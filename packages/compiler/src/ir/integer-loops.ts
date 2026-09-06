import type { IrExpr, IrLocal, IrModule, IrStmt } from "./ir.js";

/** A canonical byte loop whose induction variable is mathematically an
 * unsigned integer for every body entry. Backends may keep this binding in
 * integer storage while the loop runs, converting to f64 at ordinary JS
 * number uses and using the integer directly for typed-array indices. */
export interface IntegerBytesForLoop {
  localId: string;
  /** The `bytes.length` receiver, for the original byte-loop shape: the
   * backend compares the shadow against `receiver->len` directly. Null when
   * the limit is a general proven-integer expression (`limitExpr`). */
  limitReceiver: IrExpr | null;
  /** A proven non-negative-integer limit expression, for the general shape
   * (`i < N`, `i < N * N`). The backend re-evaluates it each iteration, as
   * JS does, in integer arithmetic. */
  limitExpr: IrExpr | null;
  /** Inclusive upper bound on the induction variable at body entry. */
  max: number;
}

/** True when a lowered subtree writes `localId`. Local ids are unique per
 * function, so a generic structured walk is sufficient and includes writes
 * nested in expressions, branches, nested loops, and try/finally bodies. */
function writesLocal(value: unknown, localId: string): boolean {
  if (Array.isArray(value)) return value.some((item) => writesLocal(item, localId));
  if (value === null || typeof value !== "object") return false;
  const node = value as { kind?: unknown; localId?: unknown };
  if (
    (node.kind === "assign" || node.kind === "assignExpr" || node.kind === "incDec") &&
    node.localId === localId
  ) {
    return true;
  }
  for (const [key, child] of Object.entries(value)) {
    if (key === "type" || key === "loc") continue;
    if (writesLocal(child, localId)) return true;
  }
  return false;
}

function isUnitIncrement(update: IrStmt | null, localId: string): boolean {
  // The frontend normally lowers `i++` to `i = i + 1` before backend
  // emission. Accept the incDec form too so this analysis remains valid for
  // hand-built IR and if normalization is moved later in the pipeline.
  if (update?.kind === "exprStmt") {
    return (
      update.expr.kind === "incDec" &&
      update.expr.localId === localId &&
      update.expr.op === "+"
    );
  }
  return (
    update?.kind === "assign" &&
    update.localId === localId &&
    update.value.kind === "bin" &&
    update.value.op === "+" &&
    update.value.left.kind === "varRef" &&
    update.value.left.localId === localId &&
    update.value.right.kind === "numLit" &&
    update.value.right.value === 1
  );
}

/** Recognize the deliberately small, semantics-transparent first tier of
 * integer induction:
 *
 *   for (let i = 0; i < bytes.length; i++) { ... }
 *
 * The binding must be an unboxed mutable f64 local and the body must not
 * write it. The exact zero start plus unit increment and ScrBytes' fixed,
 * safe-integer length prove every body value is an exactly representable
 * non-negative integer. A captured loop binding is boxed and therefore
 * refused (per-iteration binding identity remains on the generic path).
 */
export function matchIntegerBytesForLoop(
  stmt: IrStmt & { kind: "for" },
  locals: ReadonlyMap<string, IrLocal>,
): IntegerBytesForLoop | null {
  const init = stmt.init;
  if (
    init?.kind !== "varDecl" ||
    init.init?.kind !== "numLit" ||
    init.init.value !== 0 ||
    Object.is(init.init.value, -0)
  ) {
    return null;
  }
  const local = locals.get(init.localId);
  if (local?.type.kind !== "f64" || !local.mutable || local.boxed === true) return null;

  const cond = stmt.cond;
  if (
    cond?.kind !== "bin" ||
    cond.op !== "<" ||
    cond.left.kind !== "varRef" ||
    cond.left.localId !== init.localId ||
    cond.right.kind !== "bytesIntrinsic" ||
    cond.right.method !== "length" ||
    cond.right.receiver.kind !== "varRef" ||
    cond.right.receiver.type.kind !== "bytes"
  ) {
    return null;
  }
  const receiverLocal = locals.get(cond.right.receiver.localId);
  if (receiverLocal?.boxed === true) return null;

  if (!isUnitIncrement(stmt.update, init.localId)) return null;
  if (writesLocal(stmt.body, init.localId)) return null;

  return {
    localId: init.localId,
    limitReceiver: cond.right.receiver,
    limitExpr: null,
    // A ScrBytes length is bounded only by memory. That is enough for the
    // direct `bytes[i]` use this shape exists for, and too wide to survive
    // being multiplied, which the bound arithmetic then refuses on its own.
    max: INT_LIMIT - 1,
  };
}

/* ── integer index expressions ─────────────────────────────────────────── */

/** The exclusive ceiling for exact f64 integers. A value proven to sit in
 * [0, 2^53) is exactly representable as a double AND fits a uint64_t, so
 * the two representations agree and integer arithmetic below the bound
 * cannot round. Every rule here carries a static upper bound and refuses to
 * combine bounds that would leave the range. */
const INT_LIMIT = 9007199254740992;

export interface IntegerIndexHost {
  /** localId → the induction shadow standing in for the binding, with the
   * bound the loop condition proves. Backends key their own storage for
   * the shadow off the same localId. */
  readonly integerLoopBindings: ReadonlyMap<string, { max: number }>;
  /** localId → a proven non-negative integer number binding still living in
   * its ordinary f64 slot, with its static upper bound. Populated as
   * declarations are emitted. */
  readonly integerBindings: ReadonlyMap<string, { max: number }>;
}

/** A proven-integer index, as a small tree each backend renders in its own
 * integer type: the C backend as a uint64_t expression, the LLVM backend as
 * sizeType arithmetic. Kept structural rather than pre-rendered so one
 * analysis serves both. */
export type IntegerIndexNode =
  | { kind: "lit"; value: number }
  /** The loop's integer induction shadow for `localId`. */
  | { kind: "shadow"; localId: string }
  /** An ordinary f64 binding proven to hold a non-negative integer; the
   * backend converts it (one float→int instruction, no range check). */
  | { kind: "binding"; localId: string }
  | { kind: "bin"; op: "+" | "*"; left: IntegerIndexNode; right: IntegerIndexNode };

export interface IntegerIndexValue {
  node: IntegerIndexNode;
  /** Proven upper bound (inclusive) on the value. */
  max: number;
}

/** Prove `expr` is a non-negative integer below 2^53 and, if so, build the
 * equivalent uint64_t C expression.
 *
 * The point is flattened-matrix indexing (`i * N + j`): the operands are
 * induction variables and loop-invariant integer bounds, so the whole
 * address computation can stay in integers and the element accessor needs
 * a single unsigned compare instead of the negative/fractional/range
 * triple. Bounds are propagated through + and * and the rule refuses any
 * combination that could reach 2^53, which keeps the u64 arithmetic exact
 * and identical to the f64 arithmetic it replaces. Subtraction and
 * division are refused outright (they leave the non-negative integers). */
export function integerIndexExpr(expr: IrExpr, host: IntegerIndexHost): IntegerIndexValue | null {
  switch (expr.kind) {
    case "numLit": {
      const v = expr.value;
      if (!Number.isInteger(v) || Object.is(v, -0) || v < 0 || v >= INT_LIMIT) return null;
      return { node: { kind: "lit", value: v }, max: v };
    }
    case "varRef": {
      const shadow = host.integerLoopBindings.get(expr.localId);
      if (shadow !== undefined) {
        return { node: { kind: "shadow", localId: expr.localId }, max: shadow.max };
      }
      const bound = host.integerBindings.get(expr.localId);
      return bound ? { node: { kind: "binding", localId: expr.localId }, max: bound.max } : null;
    }
    case "bin": {
      if (expr.op !== "+" && expr.op !== "*") return null;
      const l = integerIndexExpr(expr.left, host);
      if (!l) return null;
      const r = integerIndexExpr(expr.right, host);
      if (!r) return null;
      const max = expr.op === "+" ? l.max + r.max : l.max * r.max;
      if (!(max < INT_LIMIT)) return null;
      return { node: { kind: "bin", op: expr.op, left: l.node, right: r.node }, max };
    }
    default:
      return null;
  }
}

/** Render a proven-integer index as a C uint64_t expression. `shadow` and
 * `binding` name the emitter's storage for a loop shadow and for an
 * ordinary f64 binding respectively. */
export function renderIntegerIndexC(
  node: IntegerIndexNode,
  shadow: (localId: string) => string,
  binding: (localId: string) => string,
): string {
  switch (node.kind) {
    case "lit":
      return `UINT64_C(${node.value})`;
    case "shadow":
      return shadow(node.localId);
    case "binding":
      return `(uint64_t)${binding(node.localId)}`;
    case "bin":
      return `(${renderIntegerIndexC(node.left, shadow, binding)} ${node.op} ${renderIntegerIndexC(node.right, shadow, binding)})`;
  }
}


/** Collect module-level `const` bindings of type number whose initializer is
 * a plain non-negative integer literal — `const N = 400`. Globals carry no
 * initializer in the IR (the per-file `%init.<i>` functions assign them), so
 * this reads the assignment back out. Immutability makes that assignment the
 * binding's only writer, which is what lets the value be treated as a
 * compile-time bound at every use. */
export function collectIntegerGlobals(mod: IrModule): Map<string, { max: number }> {
  const out = new Map<string, { max: number }>();
  const byId = new Map((mod.globals ?? []).map((g) => [g.id, g] as const));
  for (const fn of mod.functions) {
    if (!fn.name.startsWith("%init.")) continue;
    const visit = (stmts: readonly IrStmt[]): void => {
      for (const s of stmts) {
        if (s.kind !== "assign") continue;
        const g = byId.get(s.localId);
        if (!g || g.mutable || g.type.kind !== "f64") continue;
        const v = s.value;
        if (v.kind !== "numLit") continue;
        if (!Number.isInteger(v.value) || Object.is(v.value, -0) || v.value < 0 || v.value >= INT_LIMIT) {
          continue;
        }
        out.set(g.id, { max: v.value });
      }
    };
    visit(fn.body);
  }
  return out;
}

/** The general integer induction shape, a superset of
 * matchIntegerBytesForLoop:
 *
 *   for (let i = 0; i < LIMIT; i++) { ... }
 *
 * where LIMIT is either a ScrBytes length (the original rule) or any
 * expression integerIndexExpr can prove is a non-negative integer below
 * 2^53 — a numeric literal, an integer `const`, or + / * over those. The
 * exact zero start, unit increment, and a body that never writes the
 * binding then make every body value an exactly representable non-negative
 * integer strictly below LIMIT, so the loop can run on an integer shadow
 * and the bound propagates into index arithmetic (`a[i * N + j]`).
 *
 * The bound matters as much as the match: it is what lets integerIndexExpr
 * prove `i * N + j` cannot reach 2^53, and therefore that computing it in
 * uint64_t gives the same value the double arithmetic would. */
export function matchIntegerForLoop(
  stmt: IrStmt & { kind: "for" },
  locals: ReadonlyMap<string, IrLocal>,
  host: IntegerIndexHost,
): IntegerBytesForLoop | null {
  const bytes = matchIntegerBytesForLoop(stmt, locals);
  if (bytes) return bytes;

  const init = stmt.init;
  if (
    init?.kind !== "varDecl" ||
    init.init?.kind !== "numLit" ||
    init.init.value !== 0 ||
    Object.is(init.init.value, -0)
  ) {
    return null;
  }
  const local = locals.get(init.localId);
  if (local?.type.kind !== "f64" || !local.mutable || local.boxed === true) return null;

  const cond = stmt.cond;
  if (
    cond?.kind !== "bin" ||
    cond.op !== "<" ||
    cond.left.kind !== "varRef" ||
    cond.left.localId !== init.localId
  ) {
    return null;
  }
  // The limit is re-read every iteration, so it must not depend on the
  // induction variable itself — an integer `const` or literal cannot.
  const limit = integerIndexExpr(cond.right, host);
  if (!limit || limit.max === 0) return null;
  // A limit that reads the induction variable would make the bound
  // circular; only literals and immutable bindings get this far, and
  // neither can.

  if (!isUnitIncrement(stmt.update, init.localId)) return null;
  if (writesLocal(stmt.body, init.localId)) return null;

  return { localId: init.localId, limitReceiver: null, limitExpr: cond.right, max: limit.max - 1 };
}

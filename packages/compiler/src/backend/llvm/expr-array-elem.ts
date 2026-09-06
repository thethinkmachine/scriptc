/* Inline ScrArr element access for the LLVM backend — the counterpart of
 * the C backend's arrElementHelpers, and shaped after emitBytesIndex.
 *
 * A `call @scr_arr_get_f64` is opaque to the optimizer: it cannot hoist
 * `a->data`/`a->len` out of a loop, cannot keep either in a register, and
 * must treat every element access as a full memory clobber, which also
 * blocks vectorization of the surrounding loop. Emitting the checked load
 * and store directly restores all of that. The cold out-of-range path
 * still calls the runtime trap, so the RangeError text is byte-identical
 * to the out-of-line accessor's, and ref elements keep the runtime call —
 * their retain and unlink-then-release discipline belongs with the
 * collector, not inlined here. */
import type { LlvmEmitterContext, LlValue } from "./expr-context.js";
import { f64Lit } from "./common.js";
import { BOOL, F64, type IrExpr } from "../../ir/ir.js";
import { integerIndexExpr, type IntegerIndexNode } from "../../ir/integer-loops.js";

/** ScrArr field indices (see the struct in scr_runtime.h). */
const ARR_LEN = 1;
const ARR_DATA = 7;

function arrLen(host: LlvmEmitterContext, receiver: string): string {
  const B = host.B;
  const p = B.tmp();
  const len = B.tmp();
  B.line(`${p} = getelementptr inbounds %ScrArr, ptr ${receiver}, i64 0, i32 ${ARR_LEN}`);
  B.line(`${len} = load ${host.sizeType}, ptr ${p}`);
  return len;
}

function arrData(host: LlvmEmitterContext, receiver: string): string {
  const B = host.B;
  const p = B.tmp();
  const data = B.tmp();
  B.line(`${p} = getelementptr inbounds %ScrArr, ptr ${receiver}, i64 0, i32 ${ARR_DATA}`);
  B.line(`${data} = load ptr, ptr ${p}`);
  return data;
}

function trapIndex(host: LlvmEmitterContext, index: string, len: string, f64Index: boolean): void {
  const B = host.B;
  let asF64 = index;
  if (!f64Index) {
    asF64 = B.tmp();
    B.line(`${asF64} = uitofp ${host.sizeType} ${index} to double`);
  }
  host.declare(`declare void @scr_arr_trap_index(double, ${host.sizeType}) noreturn`);
  B.line(`call void @scr_arr_trap_index(double ${asF64}, ${host.sizeType} ${len})`);
  B.terminate("unreachable");
}

/** Validate `index` against the array and return it as a sizeType value.
 * `limit` is the length for reads and length + 1 for writes (i == len
 * appends). Bounds are proven BEFORE the float→int conversion: fptoui of a
 * negative, NaN, or out-of-range double is poison, so the range test has
 * to dominate it. The round trip back to double then rejects fractional
 * indices without a call to trunc. */
export function emitArrIndex(
  host: LlvmEmitterContext,
  receiver: string,
  index: string,
  opts: { integerIndex: boolean; allowAppend: boolean },
): { idx: string; len: string } {
  const B = host.B;
  const len = arrLen(host, receiver);
  let limit = len;
  if (opts.allowAppend) {
    limit = B.tmp();
    B.line(`${limit} = add ${host.sizeType} ${len}, 1`);
  }
  if (opts.integerIndex) {
    const inRange = B.tmp();
    B.line(`${inRange} = icmp ult ${host.sizeType} ${index}, ${limit}`);
    const invalid = B.newLabel("arr.index.invalid");
    const valid = B.newLabel("arr.index.valid");
    B.condBr(inRange, valid, invalid);
    B.startBlock(invalid);
    trapIndex(host, index, len, false);
    B.startBlock(valid);
    return { idx: index, len };
  }
  const limitF64 = B.tmp();
  const nonnegative = B.tmp();
  const belowLimit = B.tmp();
  const inRange = B.tmp();
  B.line(`${limitF64} = uitofp ${host.sizeType} ${limit} to double`);
  B.line(`${nonnegative} = fcmp oge double ${index}, ${f64Lit(0)}`);
  B.line(`${belowLimit} = fcmp olt double ${index}, ${limitF64}`);
  B.line(`${inRange} = and i1 ${nonnegative}, ${belowLimit}`);

  const rangeOk = B.newLabel("arr.index.range");
  const invalid = B.newLabel("arr.index.invalid");
  const valid = B.newLabel("arr.index.valid");
  B.condBr(inRange, rangeOk, invalid);

  B.startBlock(rangeOk);
  const idx = B.tmp();
  const roundTrip = B.tmp();
  const integral = B.tmp();
  B.line(`${idx} = fptoui double ${index} to ${host.sizeType}`);
  B.line(`${roundTrip} = uitofp ${host.sizeType} ${idx} to double`);
  B.line(`${integral} = fcmp oeq double ${roundTrip}, ${index}`);
  B.condBr(integral, valid, invalid);

  B.startBlock(invalid);
  trapIndex(host, index, len, true);

  B.startBlock(valid);
  return { idx, len };
}

/** An f64 or bool element read. Slots are 8 bytes wide and an f64 slot
 * holds the double's own bits, so the double loads straight out. */
export function emitArrGetScalar(
  host: LlvmEmitterContext,
  acc: "f64" | "bool",
  receiver: string,
  index: string,
  integerIndex: boolean,
): LlValue {
  const B = host.B;
  const { idx } = emitArrIndex(host, receiver, index, { integerIndex, allowAppend: false });
  const data = arrData(host, receiver);
  const p = B.tmp();
  const out = B.tmp();
  if (acc === "f64") {
    B.line(`${p} = getelementptr inbounds double, ptr ${data}, ${host.sizeType} ${idx}`);
    B.line(`${out} = load double, ptr ${p}`);
    return { name: out, type: F64 };
  }
  const raw = B.tmp();
  B.line(`${p} = getelementptr inbounds i64, ptr ${data}, ${host.sizeType} ${idx}`);
  B.line(`${raw} = load i64, ptr ${p}`);
  B.line(`${out} = icmp ne i64 ${raw}, 0`);
  return { name: out, type: BOOL };
}

/** An f64 or bool element write. `i == len` appends, which may grow the
 * backing store — that branch stays in the runtime, out of the loop body. */
export function emitArrSetScalar(
  host: LlvmEmitterContext,
  acc: "f64" | "bool",
  receiver: string,
  index: string,
  value: string,
  integerIndex: boolean,
): void {
  const B = host.B;
  const { idx, len } = emitArrIndex(host, receiver, index, { integerIndex, allowAppend: true });
  const argTy = acc === "f64" ? "double" : "i1";
  const appending = B.tmp();
  B.line(`${appending} = icmp eq ${host.sizeType} ${idx}, ${len}`);
  const append = B.newLabel("arr.set.append");
  const store = B.newLabel("arr.set.store");
  const done = B.newLabel("arr.set.done");
  B.condBr(appending, append, store);

  B.startBlock(append);
  const idxF64 = B.tmp();
  B.line(`${idxF64} = uitofp ${host.sizeType} ${idx} to double`);
  host.declare(`declare void @scr_arr_set_${acc}(ptr, double, ${acc === "bool" ? "i1 zeroext" : argTy})`);
  B.line(`call void @scr_arr_set_${acc}(ptr ${receiver}, double ${idxF64}, ${argTy} ${value})`);
  B.br(done);

  B.startBlock(store);
  const data = arrData(host, receiver);
  const p = B.tmp();
  if (acc === "f64") {
    B.line(`${p} = getelementptr inbounds double, ptr ${data}, ${host.sizeType} ${idx}`);
    B.line(`store double ${value}, ptr ${p}`);
  } else {
    const wide = B.tmp();
    B.line(`${wide} = zext i1 ${value} to i64`);
    B.line(`${p} = getelementptr inbounds i64, ptr ${data}, ${host.sizeType} ${idx}`);
    B.line(`store i64 ${wide}, ptr ${p}`);
  }
  B.br(done);

  B.startBlock(done);
}

/** Render a proven-integer index (ir/integer-loops.ts) as sizeType
 * arithmetic, or null when the index is not provably an integer. Induction
 * shadows are already integers; any other proven binding is converted with
 * a single fptoui — sound with no range check precisely because the bound
 * analysis established 0 <= value < 2^53. */
export function emitIntegerIndex(host: LlvmEmitterContext, expr: IrExpr): string | null {
  const value = integerIndexExpr(expr, {
    integerLoopBindings: host.integerLoopBindings,
    integerBindings: host.integerBindings,
  });
  return value === null ? null : renderIntegerIndex(host, value.node);
}

export function renderIntegerIndex(host: LlvmEmitterContext, node: IntegerIndexNode): string {
  const B = host.B;
  switch (node.kind) {
    case "lit":
      return String(node.value);
    case "shadow": {
      const out = B.tmp();
      B.line(`${out} = load ${host.sizeType}, ptr ${host.integerLoopBindings.get(node.localId)!.slot}`);
      return out;
    }
    case "binding": {
      const b = host.binding(node.localId);
      const raw = B.tmp();
      const out = B.tmp();
      B.line(`${raw} = load double, ptr ${b.slot}`);
      B.line(`${out} = fptoui double ${raw} to ${host.sizeType}`);
      return out;
    }
    case "bin": {
      const l = renderIntegerIndex(host, node.left);
      const r = renderIntegerIndex(host, node.right);
      const out = B.tmp();
      B.line(`${out} = ${node.op === "+" ? "add" : "mul"} ${host.sizeType} ${l}, ${r}`);
      return out;
    }
  }
}

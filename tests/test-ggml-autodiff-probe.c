/* Minimal reproduction of test-opt's test_grad(), instrumented to print the NUMBERS.
 *
 * test_grad() only prints OK/FAIL. It builds exactly this graph:
 *     weights      : 1 element, ggml_set_param
 *     intermediary = ggml_add(inputs, weights)
 *     outputs      = ggml_scale(intermediary, 1.0f)
 *     loss         = SUM(outputs)
 * so d(loss)/d(weights) is exactly 1.0 per evaluation, and ggml_opt_grad_acc accumulates, giving
 * idata+1 after the (idata+1)-th eval. It reports FAIL and says nothing about WHY.
 *
 * This prints the accumulated gradient after every eval, so the failure has a shape:
 *   all 0.0        -> the backward pass produces nothing, or grad_acc is never written
 *   1,1,1,1        -> the gradient is right but ACCUMULATION is broken (reset each eval)
 *   2,4,6,8        -> accumulating twice
 *   anything else  -> the ADD or SCALE backward is numerically wrong
 *
 * It also builds a SECOND graph without the ggml_scale, isolating ADD from SCALE. SCALE is one of
 * the four items hand-decided during the autodiff port -- the fork's ggml_scale_impl takes a bias
 * argument upstream's does not -- so separating them is the first thing worth knowing.
 *
 * CONTROL: the FORWARD value is printed too. If the forward is wrong the graph is not what this
 * comment claims and no conclusion about the backward follows.
 */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
// ggml_graph_compute_with_ctx and ggml_set_f32/ggml_get_f32_1d live here, not in ggml.h -- the
// no-scheduler bisection control needs them.
#include "ggml-cpu.h"

#include <stdio.h>
#include <string.h>

static void run_case(const char * label, int use_scale) {
    printf("\n=== %s ===\n", label);

    struct ggml_init_params ps = { 1024*1024, NULL, true };
    struct ggml_context * ctx_static  = ggml_init(ps);
    struct ggml_context * ctx_compute = ggml_init(ps);

    struct ggml_tensor * inputs  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(inputs, "inputs");
    struct ggml_tensor * weights = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(weights, "weights");
    ggml_set_param(weights);

    struct ggml_tensor * mid = ggml_add(ctx_compute, inputs, weights);
    struct ggml_tensor * outputs = use_scale ? ggml_scale(ctx_compute, mid, 1.0f) : mid;
    ggml_set_name(outputs, "outputs");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) { printf("  no CPU backend -- cannot run\n"); return; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    if (!buf) { printf("  alloc failed\n"); return; }

    const float w0 = 0.0f;
    ggml_backend_tensor_set(weights, &w0, 0, sizeof(float));

    ggml_backend_t bes[1] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, NULL, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    struct ggml_opt_params op = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    op.ctx_compute     = ctx_compute;
    op.inputs          = inputs;
    op.outputs         = outputs;
    // MEASURED: with the default opt_period of 1, ggml_opt_grad_acc(opt, weights) returns NULL and
    // this probe printed its own -12345 sentinel six times. That is not an engine defect -- it is
    // ggml-opt.cpp's own rule at :465, where a grad accumulator is created for a PARAM node only
    // when `accumulate` is true, and `accumulate` follows opt_period. test_grad passes
    // nbatch_logical=999999 with nbatch_physical=1, so its opt_period is 999999: gradients
    // accumulate for the whole run and the optimizer step never fires, which is the point of the
    // test. Reproducing the graph without reproducing this made the probe measure nothing.
    op.opt_period      = 999999;
    // Do NOT set build_type. tests/test-opt.cpp never sets it either -- helper_get_ctx_data() takes
    // whatever ggml_opt_default_params() gives, and a faithful reproduction has to do the same.
    // MEASURED: setting it to GGML_OPT_BUILD_TYPE_GRAD aborts at ggml-opt.cpp:501,
    // GGML_ASSERT(build_type_alloc == GGML_OPT_BUILD_TYPE_OPT). That assert is reachable because
    // the GRAD branch twenty lines above allocates buf_static and calls ggml_graph_reset but does
    // NOT return, while the FORWARD branch in the same shape at :452 DOES return. Whether that
    // missing return is a merge loss or is upstream's own is a separate question, filed as its own
    // row; it is not this probe's subject and test-opt never reaches it.
    ggml_opt_context_t opt = ggml_opt_init(op);

    printf("  eval :   input      forward(loss)   grad_acc(weights)   expected\n");
    for (int i = 0; i < 6; ++i) {
        const float in = (float) i;
        ggml_opt_alloc(opt, /*backward =*/ true);
        ggml_backend_tensor_set(inputs, &in, 0, sizeof(float));
        ggml_opt_result_t res = ggml_opt_result_init();
        ggml_opt_eval(opt, res);

        float g = -12345.0f;
        struct ggml_tensor * ga = ggml_opt_grad_acc(opt, weights);
        if (ga) {
            ggml_backend_tensor_get(ga, &g, 0, sizeof(float));
        }
        // THE FORWARD CONTROL, second attempt. MEASURED: ggml_opt_result_loss() printed
        // -2260374524771804410022709306589184.0 -- uninitialised garbage -- because with
        // opt_period 999999 the result carries no completed optimizer period to average over.
        // A control that reports garbage is not a control, so the forward value is read from the
        // LOSS TENSOR itself (ggml-opt.h:151, "scalar tensor that contains the loss"), which is
        // what the graph actually computed this eval. Expected: loss == in + w0 == in, since
        // w0 = 0 and the loss type is SUM over a single element.
        float lt = -12345.0f;
        struct ggml_tensor * lten = ggml_opt_loss(opt);
        if (lten) {
            ggml_backend_tensor_get(lten, &lt, 0, sizeof(float));
        }

        // PLUMBING CONTROL. The loss tensor and the grad accumulator both read exactly 0, which has
        // two very different explanations: the graph did not compute, or this probe cannot read
        // tensors back at all. Reading back `inputs` -- which this probe allocated itself and wrote
        // this iteration -- separates them. If `inputs` reads back what was just written, tensor
        // readback works and a zero loss is the graph's answer; if it does not, every number in
        // this table is meaningless and nothing about the backward pass follows.
        float rb = -12345.0f, wv = -12345.0f;
        ggml_backend_tensor_get(inputs, &rb, 0, sizeof(float));
        ggml_backend_tensor_get(weights, &wv, 0, sizeof(float));

        printf("  %4d : %7.2f   %10.4f %-9s %14.6f   %8d   %-9s  | readback in=%.2f %-7s w=%.2f\n",
               i, in, lt, (lt == in) ? "(fwd ok)" : "(FWD BAD)",
               g, i + 1, (g == (float)(i + 1)) ? "ok" : "*** WRONG",
               rb, (rb == in) ? "(rb ok)" : "(RB BAD)", wv);
        ggml_opt_result_free(res);
    }

    ggml_opt_free(opt);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx_compute);
    ggml_free(ctx_static);
}

/* THE STEP CASE. The two cases above read the grad ACCUMULATOR and got 0.0 every time. Two things
 * could produce that: the backward pass computes nothing, or it computes correctly and only the
 * accumulator is unwired. Reading the accumulator cannot tell them apart, and the ggml-opt plumbing
 * was read end to end without finding a break -- ggml_graph_cpy carries grad_accs by POINTER, and
 * dup_graph's map_tensor copies `data` and `buffer` across, so the scheduled copy writes into the
 * same memory the static tensor owns. So the linkage is not obviously the fault and a different
 * instrument is needed.
 *
 * This case uses opt_period = 1, so every eval takes an optimizer step, and watches `weights`
 * instead. weights is a tensor THIS PROBE allocated and can read back reliably -- the plumbing
 * control above proves that for `inputs`, 6 of 6. If weights moves, gradients are reaching the
 * optimizer and only the accumulator is unwired. If weights never moves, the backward pass really
 * does produce nothing. Those imply completely different fixes.
 *
 * Note grad_acc is expected to be NULL here: at opt_period 1, `accumulate` is false and
 * ggml-opt.cpp:465 allocates no accumulator for a PARAM node. That is by design, not a failure, so
 * this case does not report it as one.
 */
static void run_step_case(void) {
    printf("\n=== OPTIMIZER STEP, opt_period=1 -- does `weights` move at all? ===\n");

    struct ggml_init_params ps = { 1024*1024, NULL, true };
    struct ggml_context * ctx_static  = ggml_init(ps);
    struct ggml_context * ctx_compute = ggml_init(ps);

    struct ggml_tensor * inputs  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(inputs, "inputs");
    struct ggml_tensor * weights = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(weights, "weights");
    ggml_set_param(weights);

    struct ggml_tensor * mid = ggml_add(ctx_compute, inputs, weights);
    struct ggml_tensor * outputs = ggml_scale(ctx_compute, mid, 1.0f);
    ggml_set_name(outputs, "outputs");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) { printf("  no CPU backend -- cannot run\n"); return; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    if (!buf) { printf("  alloc failed\n"); return; }

    const float w0 = 3.0f;
    ggml_backend_tensor_set(weights, &w0, 0, sizeof(float));

    ggml_backend_t bes[1] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, NULL, 1, GGML_DEFAULT_GRAPH_SIZE,
                                                        false, true);

    struct ggml_opt_params op = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    op.ctx_compute = ctx_compute;
    op.inputs      = inputs;
    op.outputs     = outputs;
    op.opt_period  = 1;
    ggml_opt_context_t opt = ggml_opt_init(op);

    /* FORWARD-ONLY FIRST. ggml_opt_eval does call ggml_backend_sched_graph_compute
     * (ggml-opt.cpp:824) and then reads opt_ctx->loss itself without aborting, so the loss tensor
     * is a real scalar F32. Yet the earlier cases read it as 0 for every input. Before concluding
     * anything about the BACKWARD, run one eval with backward=false -- build_type FORWARD, graph
     * = gf, no backward nodes involved at all. With input 4 and weight 3 a SUM loss must be 7.
     * If it is 0, the forward is the broken half and everything said about the backward so far is
     * downstream of that. */
    {
        const float in = 4.0f;
        ggml_opt_alloc(opt, /*backward =*/ false);
        ggml_backend_tensor_set(inputs, &in, 0, sizeof(float));
        ggml_opt_result_t r0 = ggml_opt_result_init();
        ggml_opt_eval(opt, r0);
        float lf = -12345.0f;
        struct ggml_tensor * lt = ggml_opt_loss(opt);
        if (lt) {
            ggml_backend_tensor_get(lt, &lf, 0, sizeof(float));
        }
        printf("  FORWARD-ONLY control: input=%.2f weight=%.2f  loss=%.4f  expected 7.0000  %s\n",
               in, w0, lf, (lf == 7.0f) ? "(fwd OK)" : "*** FORWARD IS BROKEN TOO");
        ggml_opt_result_free(r0);
    }

    printf("  starting weight = %.4f ; loss is SUM so d(loss)/d(w) = 1 per eval,\n", w0);
    printf("  so ANY working optimizer must move it DOWN every step.\n");
    printf("  eval :   input     weights AFTER step    moved?\n");
    float prev = w0;
    int moved_any = 0;
    for (int i = 0; i < 6; ++i) {
        const float in = (float) i;
        ggml_opt_alloc(opt, /*backward =*/ true);
        ggml_backend_tensor_set(inputs, &in, 0, sizeof(float));
        ggml_opt_result_t res = ggml_opt_result_init();
        ggml_opt_eval(opt, res);

        float w = -12345.0f;
        ggml_backend_tensor_get(weights, &w, 0, sizeof(float));
        const int moved = (w != prev);
        moved_any |= moved;
        printf("  %4d : %7.2f   %18.6f    %s\n", i, in, w, moved ? "yes" : "NO -- unchanged");
        prev = w;
        ggml_opt_result_free(res);
    }
    printf("\n  VERDICT: weights %s over 6 optimizer steps.\n",
           moved_any ? "MOVED -- gradients reach the optimizer; the accumulator is the unwired part"
                     : "NEVER MOVED -- the backward pass produces nothing at all");

    ggml_opt_free(opt);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx_compute);
    ggml_free(ctx_static);
}

/* THE CONTROL THIS PROBE SHOULD HAVE HAD FIRST.
 *
 * Everything above goes through ggml-opt: ggml_opt_init builds the graphs, ggml_opt_alloc makes a
 * COPY of the chosen graph with dup_graph and schedules the copy, and ggml_opt_eval calls
 * ggml_backend_sched_graph_compute on that copy. The forward-only case then read loss = 0.0000
 * where input 4 + weight 3 must give 7. That has two very different explanations and every number
 * printed above is downstream of which one is true:
 *
 *   (a) the CORE compute is broken -- ggml_add / ggml_sum / the scheduler produce nothing, in
 *       which case ggml-opt is an innocent caller and the autodiff port is not the subject at all;
 *   (b) the core is fine and ggml-opt's graph copying or scheduling loses the result.
 *
 * This case builds the same arithmetic with NO ggml-opt: ggml_add then ggml_sum, one
 * ggml_backend_sched_graph_compute, read the answer. Same backend, same scheduler type, same
 * F32 1-element tensors. 4 + 3 summed is 7 and nothing about training is involved.
 */
static void run_core_control(void) {
    printf("\n=== CORE CONTROL -- plain ggml, no ggml-opt: is ggml_add + ggml_sum computing? ===\n");

    struct ggml_init_params ps = { 1024*1024, NULL, true };
    struct ggml_context * ctx_static  = ggml_init(ps);
    struct ggml_context * ctx_compute = ggml_init(ps);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(a, "a");
    ggml_set_input(a);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(b, "b");
    ggml_set_input(b);

    struct ggml_tensor * sum = ggml_sum(ctx_compute, ggml_add(ctx_compute, a, b));
    ggml_set_name(sum, "sum");
    ggml_set_output(sum);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) { printf("  no CPU backend -- cannot run\n"); return; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    if (!buf) { printf("  alloc failed\n"); return; }

    const float av = 4.0f, bv = 3.0f;
    ggml_backend_tensor_set(a, &av, 0, sizeof(float));
    ggml_backend_tensor_set(b, &bv, 0, sizeof(float));

    ggml_backend_t bes[1] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, NULL, 1, GGML_DEFAULT_GRAPH_SIZE,
                                                        false, true);
    struct ggml_cgraph * gf = ggml_new_graph(ctx_compute);
    ggml_build_forward_expand(gf, sum);
    // THE STATUS WAS BEING THROWN AWAY. Both of these return ggml_status, and the first version of
    // this control ignored both -- so "the answer is 0" could not be told apart from "the compute
    // never ran and said so". Printing them costs nothing and removes a whole class of guessing.
    bool alloc_ok = ggml_backend_sched_alloc_graph(sched, gf);
    enum ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    printf("  sched_alloc_graph -> %s ; sched_graph_compute -> %d (%s)\n",
           alloc_ok ? "true" : "FALSE",
           (int) st, st == GGML_STATUS_SUCCESS ? "GGML_STATUS_SUCCESS" : "NOT SUCCESS");
    printf("  graph: n_nodes=%d n_leafs=%d\n", ggml_graph_n_nodes(gf), gf->n_leafs);

    float out = -12345.0f;
    ggml_backend_tensor_get(sum, &out, 0, sizeof(float));
    printf("  ggml_sum(ggml_add(%.2f, %.2f)) = %.4f   expected 7.0000   %s\n",
           av, bv, out, (out == 7.0f) ? "(CORE OK)" : "*** THE CORE COMPUTE IS BROKEN");

    // BISECTION: the SAME graph without ggml_backend_sched at all. ggml_graph_compute_with_ctx
    // runs the CPU kernels directly on tensors that own their own malloc'd data, so it shares no
    // machinery with the path above except the kernels themselves.
    //   this OK, sched broken -> the scheduler/allocator is the subject, not the kernels
    //   both broken          -> ggml_add or ggml_sum is the subject
    {
        struct ggml_init_params dp = { 16*1024*1024, NULL, false };   // no_alloc = false: real data
        struct ggml_context * dctx = ggml_init(dp);
        struct ggml_tensor * da = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, 1);
        struct ggml_tensor * db = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, 1);
        ggml_set_f32(da, 4.0f);
        ggml_set_f32(db, 3.0f);
        // PER-OP, because "the core is broken" is not yet a defect anyone can fix. The ADD result
        // is read separately from the SUM result, so the two ops are judged independently:
        //   add OK, sum 0   -> ggml_sum
        //   add 0           -> ggml_add, and sum is only reporting its input
        struct ggml_tensor * dadd = ggml_add(dctx, da, db);
        struct ggml_tensor * dsum = ggml_sum(dctx, dadd);
        struct ggml_cgraph * dgf = ggml_new_graph(dctx);
        ggml_build_forward_expand(dgf, dsum);
        // THE PLAN IS INSPECTED, NOT ASSUMED. ggml_graph_compute_with_ctx builds a ggml_cplan
        // internally and hands it to ggml_graph_compute; if that plan carries n_threads == 0, or a
        // work_size the wrapper does not satisfy, the compute loop can run no nodes at all and
        // STILL return GGML_STATUS_SUCCESS -- which is exactly the symptom being chased. Printing
        // the plan separates "the kernel computed the wrong answer" from "the kernel never ran".
        // ggml_compute_forward's FIRST statement is
        //     if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) return;
        // (ggml-cpu/ggml-cpu.c:1715). If ggml_is_empty() answers true for an ordinary tensor, every
        // op returns before dispatch, computes nothing, and the graph still reports
        // GGML_STATUS_SUCCESS -- the exact symptom. It is one call to check and it rules an entire
        // branch in or out, so it is checked rather than reasoned about.
        printf("     add tensor        : op=%s ne=[%lld,%lld,%lld,%lld] nelements=%lld "
               "is_empty=%s %s\n",
               ggml_op_name(dadd->op),
               (long long) dadd->ne[0], (long long) dadd->ne[1],
               (long long) dadd->ne[2], (long long) dadd->ne[3],
               (long long) ggml_nelements(dadd),
               ggml_is_empty(dadd) ? "TRUE" : "false",
               ggml_is_empty(dadd) ? "*** is_empty IS TRUE -> every op returns before dispatch"
                                   : "");
        {
            struct ggml_cplan plan = ggml_graph_plan(dgf, 1, NULL);
            printf("     ggml_graph_plan   : n_threads=%d work_size=%zu %s\n",
                   plan.n_threads, plan.work_size,
                   (plan.n_threads > 0) ? "" : "*** n_threads IS ZERO -- no node can run");
        }
        enum ggml_status dst = ggml_graph_compute_with_ctx(dctx, dgf, 1);
        const float dav = ggml_get_f32_1d(dadd, 0);
        const float dv = ggml_get_f32_1d(dsum, 0);
        printf("  NO-SCHEDULER control: ggml_graph_compute_with_ctx -> %d\n", (int) dst);
        printf("     inputs   read back : a=%.4f b=%.4f   %s\n",
               ggml_get_f32_1d(da, 0), ggml_get_f32_1d(db, 0),
               (ggml_get_f32_1d(da, 0) == 4.0f && ggml_get_f32_1d(db, 0) == 3.0f)
                   ? "(inputs ok)" : "*** INPUTS WRONG -- ggml_set_f32 is the subject");
        printf("     ggml_add          : %.4f  expected 7.0000  %s\n", dav,
               (dav == 7.0f) ? "(ADD OK)" : "*** ADD IS THE SUBJECT");
        printf("     ggml_sum(add)     : %.4f  expected 7.0000  %s\n", dv,
               (dv == 7.0f) ? "(SUM OK)"
                            : (dav == 7.0f ? "*** SUM IS THE SUBJECT (add was fine)"
                                           : "(downstream of add)"));
        ggml_free(dctx);
    }
    printf("  -> if the core is OK, the zero loss above is ggml-opt losing the result, not ggml.\n");
    printf("  -> if the core is BROKEN, the autodiff port is not the subject; ggml_add or the\n");
    printf("     scheduler is, and every autodiff number measured so far is downstream of it.\n");

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx_compute);
    ggml_free(ctx_static);
}

// THE PREVENTION, and the reason this file stopped being a print-only probe.
//
// This tree shipped for days in a state where EVERY graph on EVERY backend computed nothing and
// reported GGML_STATUS_SUCCESS, because ggml_visit_parents no longer set GGML_TENSOR_FLAG_COMPUTE
// while a dozen backends still tested it. Nothing failed. The build was green, the status codes
// were green, and the only symptom was numbers that were quietly zero.
//
// So the arithmetic assertion below is deliberately the crudest possible check -- ggml_add of two
// scalars -- and it EXITS NON-ZERO. A test that always exits 0 would have passed throughout the
// outage, which is exactly why the first version of this file was not registered with ctest and is
// now registered.
static int g_fail = 0;

static void must_equal(const char * what, float got, float want) {
    const bool ok = (got == want);
    printf("  ASSERT %-46s got %10.4f want %10.4f  %s\n", what, got, want,
           ok ? "ok" : "*** FAIL");
    if (!ok) {
        g_fail++;
    }
}

static void run_assertions(void) {
    printf("\n=== ASSERTIONS (this is the part that can FAIL the build) ===\n");

    struct ggml_init_params p = { 16*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(p);
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_f32(a, 4.0f);
    ggml_set_f32(b, 3.0f);
    struct ggml_tensor * add = ggml_add(ctx, a, b);
    struct ggml_tensor * sum = ggml_sum(ctx, add);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sum);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    must_equal("ggml_add(4,3)", ggml_get_f32_1d(add, 0), 7.0f);
    must_equal("ggml_sum(add)", ggml_get_f32_1d(sum, 0), 7.0f);

    // The flag whose absence caused the outage, asserted DIRECTLY rather than only through its
    // effect -- so a future regression is named rather than merely producing a wrong number.
    const bool flagged = (add->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    printf("  ASSERT %-46s %s  %s\n", "ggml_build_forward_expand sets FLAG_COMPUTE",
           flagged ? "     set" : "   UNSET", flagged ? "ok" : "*** FAIL");
    if (!flagged) {
        g_fail++;
    }

    ggml_free(ctx);
}

int main(void) {
    ggml_backend_load_all();
    printf("POSITIVE CONTROL ggml build: %s\n", "linked and running");
    run_core_control();
    run_case("WITH ggml_scale (exactly test_grad's graph)", 1);
    run_case("WITHOUT ggml_scale (isolates ADD backward)", 0);
    run_step_case();
    printf("\nReminder: grad_acc must ACCUMULATE, so the expected column is idata+1, not 1.\n");
    run_assertions();
    printf("\n%s %d assertion failure(s)\n", g_fail ? "FAILED --" : "PASS --", g_fail);
    return g_fail ? 1 : 0;
}

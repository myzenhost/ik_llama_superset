#include "server-task.h"
#include "server-queue.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <thread>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

//
// server_queue
//

static bool task_resets_idle_timer(server_task_type type) {
    return type != SERVER_TASK_TYPE_METRICS;
}

// The sleeping-state callbacks run on the loop thread with mutex_tasks held (start_loop). A queue method called from inside one
// would lock mutex_tasks again on the same thread and block forever with no log line (2026-10-17: the server never answered after
// waking from --sleep-idle-seconds because load_model called set_sched_policy). Abort with the method's name instead.
void server_queue::assert_not_in_state_callback(const char * fn) const {
    if (in_state_callback.load(std::memory_order_acquire) && state_callback_thread == std::this_thread::get_id()) {
        GGML_ABORT("server_queue::%s called from a sleeping-state callback while mutex_tasks is held (self-deadlock)", fn);
    }
}

int server_queue::post(server_task && task, bool front) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(task.id != -1);
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL) {
        cleanup_pending_task(task.id_target);
    }
    const int  task_id     = task.id;
    const bool reset_timer = task_resets_idle_timer(task.type);
    QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
    if (front) {
        queue_tasks.push_front(std::move(task));
    } else {
        queue_tasks.push_back(std::move(task));
    }
    if (reset_timer) {
        time_last_task = ggml_time_ms();
    }
    condition_tasks.notify_one();
    return task_id;
}

int server_queue::post(std::vector<server_task> && tasks, bool front) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    bool reset_timer = false;
    for (auto & task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        // if this is cancel task make sure to clean up pending tasks
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            cleanup_pending_task(task.id_target);
        }
        reset_timer |= task_resets_idle_timer(task.type);
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
    }
    if (reset_timer) {
        time_last_task = ggml_time_ms();
    }
    condition_tasks.notify_one();
    return 0;
}

void server_queue::defer(server_task && task) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    QUE_DBG("defer task, id = %d\n", task.id);
    if (task.t_first_deferred_ms == 0) {
        task.t_first_deferred_ms = ggml_time_ms(); // kept across re-deferrals: the wait is counted from the first one
        task.sched_arrival_tokens = sched_prefill_tokens.load(std::memory_order_relaxed); // hrrn age starts here, same rule
    }
    queue_tasks_deferred.push_back(std::move(task));
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

std::vector<server_queue::dropped_task> server_queue::take_expired_deferred(int64_t now_ms, int64_t now_us) {
    std::vector<dropped_task> dropped;
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ) {
        const int32_t limit = it->params.max_queue_wait_ms;
        if (it->t_deadline_us > 0 && now_us >= it->t_deadline_us) {
            // the end-to-end budget wins over the queue-wait limit: it is the stricter statement about the same request
            QUE_DBG("deferred task %d is past its deadline by %lld us, dropping\n", it->id, (long long) (now_us - it->t_deadline_us));
            dropped.push_back({it->id, DROP_DEADLINE});
            it = queue_tasks_deferred.erase(it);
        } else if (limit > 0 && it->t_first_deferred_ms > 0 && now_ms - it->t_first_deferred_ms >= limit) {
            QUE_DBG("deferred task %d waited %lld ms (limit %d), dropping\n", it->id, (long long) (now_ms - it->t_first_deferred_ms), (int) limit);
            dropped.push_back({it->id, DROP_QUEUE_WAIT});
            it = queue_tasks_deferred.erase(it);
        } else {
            ++it;
        }
    }
    return dropped;
}

int server_queue::evict_deferred(int32_t new_priority) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    // Candidates: generation tasks that have no slot yet. That is the deferred list AND the inbox `queue_tasks`: a request admitted a
    // few milliseconds ago is still in the inbox until the main loop's next iteration defers it (found by test, 2026-10-16: the
    // arrival raced the loop and saw an empty deferred list).
    std::deque<server_task> * victim_q = nullptr;
    std::deque<server_task>::iterator victim;
    auto consider = [&](std::deque<server_task> & q) {
        for (auto it = q.begin(); it != q.end(); ++it) {
            if (it->type != SERVER_TASK_TYPE_COMPLETION && it->type != SERVER_TASK_TYPE_INFILL) {
                continue; // a slot save, a metrics read, a cancel ... is not a generation request
            }
            if (victim_q == nullptr || it->params.priority > victim->params.priority ||
                (it->params.priority == victim->params.priority && it->t_submit_us >= victim->t_submit_us)) {
                victim_q = &q;
                victim   = it; // lowest importance first; among equals the newest arrival (it has waited the least)
            }
        }
    };
    consider(queue_tasks_deferred);
    consider(queue_tasks);
    if (victim_q == nullptr || victim->params.priority <= new_priority) {
        return -1;
    }
    const int id = victim->id;
    QUE_DBG("evicting queued task %d (priority %d) for an arrival of priority %d\n", id, (int) victim->params.priority, (int) new_priority);
    victim_q->erase(victim);
    return id;
}

int server_queue::shed_oldest_deferred(int64_t now_us, int64_t n_ctx_slot, const std::function<bool(double, int64_t, size_t)> & should_drop) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (queue_tasks_deferred.empty()) {
        return -1;
    }
    auto oldest = queue_tasks_deferred.begin();
    for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ++it) {
        if (it->t_submit_us < oldest->t_submit_us) {
            oldest = it;
        }
    }
    const int64_t t_arrival = oldest->t_submit_us > 0 ? oldest->t_submit_us : now_us;
    const double  wait_ms   = (double) (now_us - t_arrival) / 1000.0;
    // a request that sets no n_predict may run to the end of its slot's context
    const int64_t tokens    = oldest->params.n_predict > 0 ? (int64_t) oldest->n_tokens() + oldest->params.n_predict
                                                           : std::max<int64_t>((int64_t) oldest->n_tokens(), n_ctx_slot);
    if (!should_drop(wait_ms, tokens, queue_tasks_deferred.size())) {
        return -1;
    }
    const int id = oldest->id;
    QUE_DBG("shedding deferred task %d: waited %.0f ms, %lld tokens, %zu queued\n", id, wait_ms, (long long) tokens, queue_tasks_deferred.size());
    queue_tasks_deferred.erase(oldest);
    return id;
}

int server_queue::get_new_id() {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    return new_id;
}

void server_queue::set_sched_policy(const server_sched_params & params, std::function<size_t(const server_task &)> cached_prefix) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    sched = params;
    sched_cached_prefix = std::move(cached_prefix);
}

void server_queue::pop_deferred_task(int id_slot) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!queue_tasks_deferred.empty()) {
        // try to find a task that uses the specified slot
        bool found = false;
        for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ++it) {
            if (it->id_slot == id_slot) {
                QUE_DBG("pop deferred task (use slot %d), id_task = %d\n", id_slot, it->id);
                queue_tasks.emplace_front(std::move(*it));
                queue_tasks_deferred.erase(it);
                found = true;
                break;
            }
        }
        // if no task asked for this slot, pop the deferred task chosen by the scheduling policy: with fcfs (default) the one with the
        // lowest `priority` value, among equal values the earliest deferred one (so with every priority at its default 0 this is the
        // first task, as before); with lpm see server-sched-policy.h
        if (!found) {
            std::vector<server_sched_cand> cands;
            cands.reserve(queue_tasks_deferred.size());
            std::vector<std::vector<int32_t>> prompts; // lpm only: the in-batch comparison reads token ids
            const bool lpm = sched.policy != server_sched_policy::FCFS && queue_tasks_deferred.size() >= 2 &&
                             queue_tasks_deferred.size() <= sched.max_queue && sched_cached_prefix;
            if (lpm) {
                prompts.reserve(queue_tasks_deferred.size());
            }
            const uint64_t prefill_now = sched_prefill_tokens.load(std::memory_order_relaxed);
            for (const auto & t : queue_tasks_deferred) {
                server_sched_cand c;
                c.priority = t.params.priority;
                if (lpm) {
                    c.cached_prefix = sched_cached_prefix(t);
                    c.n_tokens      = t.tokens.size();
                    c.waited_tokens = prefill_now > t.sched_arrival_tokens ? prefill_now - t.sched_arrival_tokens : 0;
                    if (t.tokens.has_mtmd) {
                        prompts.emplace_back(); // media prompts are not compared by token
                    } else {
                        prompts.emplace_back(t.tokens.get_tokens().begin(), t.tokens.get_tokens().end());
                    }
                }
                cands.push_back(c);
            }
            if (lpm) {
                for (size_t i = 0; i < cands.size(); ++i) {
                    cands[i].tokens = &prompts[i];
                }
            }
            server_sched_params sp = sched;
            if (!lpm) {
                sp.policy = server_sched_policy::FCFS;
            }
            const server_sched_result res = server_sched_pick(sp, cands);
            auto it_pick = queue_tasks_deferred.begin() + res.pick;
            QUE_DBG("pop deferred task, id_task = %d, priority = %d\n", it_pick->id, (int) it_pick->params.priority);
            if (res.used_lpm) {
                // always on (INFO): one line per decision under --sched-policy lpm|spf|hrrn, the proof the policy ran and whether it changed the order
                QUE_INF("sched-policy %s: started task %d (cached prefix %zu token(s)) among %zu waiting; fcfs would have started task %d%s, "
                        "%zu request(s) deprioritized behind an earlier one sharing their prefix\n",
                        server_sched_policy_name(sp.policy), it_pick->id, cands[res.pick].cached_prefix, cands.size(), (queue_tasks_deferred.begin() + res.fcfs_pick)->id,
                        res.pick != res.fcfs_pick ? " (REORDERED)" : " (same)", res.n_deprioritized);
            }
            queue_tasks.emplace_front(std::move(*it_pick));
            queue_tasks_deferred.erase(it_pick);
        }
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

size_t server_queue::peek_waiting(size_t n_max, const std::function<void(const server_task &)> & fn) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    size_t n = 0;
    for (const auto * q : { &queue_tasks, &queue_tasks_deferred }) {
        for (const auto & t : *q) {
            if (n >= n_max) {
                return n;
            }
            fn(t);
            ++n;
        }
    }
    return n;
}

void server_queue::wait_until_no_sleep() {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!sleeping) {
        return;
    } else {
        if (!req_stop_sleeping) {
            QUE_DBG("%s", "requesting to stop sleeping\n");
            req_stop_sleeping = true;
            condition_tasks.notify_one(); // only main thread is waiting on this
        }
        QUE_DBG("%s", "waiting until no sleep\n");
        condition_tasks.wait(lock, [&]{
            return !sleeping;
        });
    }
}

bool server_queue::request_sleep(int64_t grace_ms) {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!sleep_enabled) {
        return false;
    }
    if (sleeping) {
        return true;
    }
    force_sleep          = true;
    force_sleep_until_ms = ggml_time_ms() + grace_ms;
    condition_tasks.notify_all();
    return true;
}

void server_queue::terminate() {
    assert_not_in_state_callback(__func__);
    std::unique_lock<std::mutex> lock(mutex_tasks);
    running = false;
    if (!loop_active) {
        terminate_pending = true; // no loop to stop yet: the next start_loop() must return at once
    }
    condition_tasks.notify_all();
}

bool server_queue::process_new_tasks(bool is_yielding) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (!running) {
            QUE_DBG("%s", "terminate\n");
            return true;
        }
        if (queue_tasks.empty()) {
            return false;
        }
        server_task task = std::move(queue_tasks.front());
        queue_tasks.pop_front();
        lock.unlock();

        QUE_DBG("processing task, id = %d\n", task.id);
        if (!callback_new_task(std::move(task), is_yielding)) {
            // set it aside, do not put it back in the queue, else we offer it again in a loop
            GGML_ASSERT(is_yielding && "a task can only be declined while yielding");
            QUE_DBG("task declined, id = %d\n", task.id);
            lock.lock();
            queue_tasks_unhandled.push_back(std::move(task));
        }
    }
}

void server_queue::worker_loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            // wait on busy instead of yielding - busy stays set even when the yield already ended
            worker.cv.wait(lock, [&]{
                return worker.stop || worker.busy;
            });
            if (worker.stop) {
                return;
            }
        }

        // process tasks while the yield is active
        while (true) {
            bool terminated = false;
            try {
                // note: do not hold any lock here, the callback may post new tasks
                terminated = process_new_tasks(true);
            } catch (...) {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                worker.exception = std::current_exception();
                break;
            }

            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (terminated || worker.stop || !worker.yielding) {
                break;
            }
            if (!queue_tasks.empty()) {
                continue; // a new task arrived in the meantime
            }
            condition_tasks.wait(lock, [&]{
                return worker.stop || !running || !worker.yielding || !queue_tasks.empty();
            });
        }

        // signal to yield_to_queue() that no more tasks will be processed
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            worker.busy = false;
        }
        condition_tasks.notify_all();
    }
}

void server_queue::worker_stop() {
    if (!worker.thread.joinable()) {
        return;
    }
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        worker.stop = true;
    }
    worker.cv.notify_one();
    condition_tasks.notify_all();
    worker.thread.join();
}

void server_queue::yield_to_queue(std::function<void()> && work) {
    GGML_ASSERT(worker.thread.joinable() && "yield_to_queue() requires start_loop() to be running");

    QUE_DBG("%s", "yielding to queue\n");

    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        GGML_ASSERT(!worker.busy && "yield_to_queue() cannot be nested");
        worker.busy     = true;
        worker.yielding = true;
    }
    worker.cv.notify_one();

    // run the work on the current thread, so that all ggml compute stays on the same thread
    std::exception_ptr exception;
    try {
        work();
    } catch (...) {
        exception = std::current_exception();
    }

    {
        std::unique_lock<std::mutex> lock(mutex_tasks);

        // the yield is over, wait for the worker to finish its current task
        worker.yielding = false;
        condition_tasks.notify_all();
        condition_tasks.wait(lock, [&]{
            return !worker.busy;
        });

        // put the declined tasks back, keeping their order
        while (!queue_tasks_unhandled.empty()) {
            queue_tasks.push_front(std::move(queue_tasks_unhandled.back()));
            queue_tasks_unhandled.pop_back();
        }

        // make sure to avoid idle timeout here
        time_last_task = ggml_time_ms();

        // an exception from work() takes precedence over the one from the worker
        if (!exception) {
            std::swap(exception, worker.exception);
        } else {
            worker.exception = nullptr;
        }
    }

    QUE_DBG("%s", "done yielding to queue\n");

    // note: rethrow only after the declined tasks are back in the queue, so they are not lost
    if (exception) {
        std::rethrow_exception(exception);
    }
}

void server_queue::start_loop(int64_t idle_sleep_ms) {
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        loop_active       = true;
        running           = !terminate_pending; // a terminate() issued before this call is honoured, not overwritten
        terminate_pending = false;
        time_last_task    = ggml_time_ms();
        sleep_enabled     = idle_sleep_ms >= 0;
        force_sleep       = false;
    }

    // spawn the worker thread used by yield_to_queue()
    GGML_ASSERT(!worker.thread.joinable() && "start_loop() is already running");
    worker.stop     = false;
    worker.busy     = false;
    worker.yielding = false;
    worker.thread = std::thread([this]() { worker_loop(); });

    constexpr auto max_wait_time = std::chrono::seconds(1);
    auto should_sleep = [&]() -> bool {
        // caller must hold mutex_tasks
        if (idle_sleep_ms < 0) {
            return false;
        }
        int64_t now = ggml_time_ms();
        if (force_sleep) {
            if (now >= force_sleep_until_ms) {
                force_sleep = false; // the request that asked for it can no longer be served by sleeping now
            } else if (!callback_is_busy || !callback_is_busy()) {
                return true; // keep_alive 0: no slot is working, do not wait out the idle timer
            }
        }
        return (now - time_last_task) >= idle_sleep_ms;
    };

    while (true) {
        QUE_DBG("%s", "processing new tasks\n");
        if (process_new_tasks(false)) {
            break; // terminate
        }

        // all tasks in the current loop is processed, slots data is now ready
        QUE_DBG("%s", "update slots\n");

        // this will run the main inference process for all slots
        const int64_t t_update_slots = ggml_time_ms();
        callback_update_slots();
        {
            // update_slots() may take a while to finish, we need to make sure it's not counted as idle
            // shift instead of reset, so that non-task_resets_idle_timer tasks do not delay the sleep
            std::unique_lock<std::mutex> lock(mutex_tasks);
            const int64_t now = ggml_time_ms();
            time_last_task = std::min(now, time_last_task + (now - t_update_slots));
        }

        QUE_DBG("%s", "waiting for new tasks\n");
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running || !queue_tasks.empty()) {
                break; // go back to process new tasks or terminate
            }

            // no tasks, check for sleeping state
            if (should_sleep()) {
                QUE_INF("%s", "entering sleeping state\n");
                sleeping = true;
                force_sleep = false;
                // Call order cb0 -> cb1 -> cb{N}
                state_callback_thread = std::this_thread::get_id();
                in_state_callback.store(true, std::memory_order_release);
                for (auto & cb : callback_sleeping_state) {
                    cb(true);
                }
                in_state_callback.store(false, std::memory_order_release);
                req_stop_sleeping = false;
                // wait until we are requested to exit sleeping state
                condition_tasks.wait(lock, [&]{
                    return (!running || req_stop_sleeping);
                });
                if (!running) { // may changed during sleep
                    break; // terminate
                }
                QUE_INF("%s", "exiting sleeping state\n");
                req_stop_sleeping = false;
                // Call order cb{N} -> cb1 -> cb0
                state_callback_thread = std::this_thread::get_id();
                in_state_callback.store(true, std::memory_order_release);
                for (size_t i = callback_sleeping_state.size(); i > 0; i--) {
                    callback_sleeping_state[i - 1](false);
                }
                in_state_callback.store(false, std::memory_order_release);
                sleeping = false;
                time_last_task = ggml_time_ms();
                condition_tasks.notify_all(); // notify wait_until_no_sleep()
                break; // process new tasks
            } else {
                // wait for new tasks or timeout for checking sleeping condition
                bool res = condition_tasks.wait_for(lock, force_sleep ? std::chrono::milliseconds(100) : std::chrono::milliseconds(max_wait_time), [&]{
                    return (!queue_tasks.empty() || !running);
                });
                if (res) {
                    break; // new task arrived or terminate
                }
                // a keep_alive 0 latched during this wait is noticed on the next pass (at most max_wait_time later), then polled every 100 ms
                // otherwise, loop again to check sleeping condition
            }
        }
    }

    worker_stop();

    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        loop_active = false;
    }
}

void server_queue::cleanup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    auto rm_func = [id_target](const server_task & task) {
        return task.id == id_target;
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(),           queue_tasks.end(),           rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(),  queue_tasks_deferred.end(),  rm_func),
        queue_tasks_deferred.end());
    // a task declined while yielding is not in queue_tasks yet, but it can still be cancelled
    queue_tasks_unhandled.erase(
        std::remove_if(queue_tasks_unhandled.begin(), queue_tasks_unhandled.end(), rm_func),
        queue_tasks_unhandled.end());
}

//
// server_response
//

void server_response::add_waiting_task_id(int id_task) {
    RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::add_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.insert(id_task);
    }
}

void server_response::remove_waiting_task_id(int id_task) {
    RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
    // make sure to clean up all pending results
    queue_results.erase(
        std::remove_if(queue_results.begin(), queue_results.end(), [id_task](const server_task_result_ptr & res) {
            return res->id == id_task;
        }),
        queue_results.end());
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

server_task_result_ptr server_response::recv(const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&]{
            if (!running) {
                RES_DBG("%s : queue result stop\n", "recv");
                std::terminate(); // we cannot return here since the caller is HTTP code
            }
            return !queue_results.empty();
        });

        for (size_t i = 0; i < queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (int i = 0; i < (int) queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_until(lock, deadline);
        if (!running) {
            RES_DBG("%s : queue result stop\n", __func__);
            std::terminate(); // we cannot return here since the caller is HTTP code
        }
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv(int id_task) {
    std::unordered_set<int> id_tasks = {id_task};
    return recv(id_tasks);
}

void server_response::send(server_task_result_ptr && result) {
    RES_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto & id_task : waiting_task_ids) {
        if (result->id == id_task) {
            RES_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}

void server_response::broadcast(server_task_result_ptr && result) {
    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto & id_task : waiting_task_ids) {
        RES_DBG("task id = %d pushed to result queue\n", id_task);
        server_task_result_ptr res_copy(result->clone());
        res_copy->id = id_task; // override id with target task id
        queue_results.emplace_back(std::move(res_copy));
    }
    condition_results.notify_all();
}

void server_response::terminate() {
    running = false;
    condition_results.notify_all();
}

//
// server_response_reader
//

void server_response_reader::post_task(server_task && task, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
    GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
    task.index = 0;
    id_tasks.insert(task.id);
    states.push_back(task.create_state());
    queue_results.add_waiting_task_id(task.id);
    queue_tasks.post(std::move(task), front);
}

void server_response_reader::post_tasks(std::vector<server_task> && tasks, const common_chat_session & session, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_tasks() can only be called once per reader");
    id_tasks = server_task::get_list_id(tasks);
    states.reserve(tasks.size());
    size_t index = 0;
    for (auto & task : tasks) {
        task.index = index++;
        states.push_back(task.create_state(session));
        // for child tasks
        for (auto & child_task : task.child_tasks) {
            child_task.index = index++;
            states.push_back(child_task.create_state(session));
        }
    }
    GGML_ASSERT(states.size() == id_tasks.size());
    queue_results.add_waiting_task_ids(id_tasks);
    queue_tasks.post(std::move(tasks), front);
}

bool server_response_reader::has_next() const {
    return !cancelled && received_count < id_tasks.size();
}

// return nullptr if should_stop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::next(const std::function<bool()> & should_stop) {
    while (true) {
        server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
        if (result == nullptr) {
            // timeout, check stop condition
            if (should_stop()) {
                return nullptr;
            }
        } else {
            if (result->is_error()) {
                stop(); // cancel remaining tasks
                SRV_DBG("%s", "received error result, stopping further processing\n");
                return result;
            }
            if (!states.empty()) {
                // update the generation state if needed
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]);
            }
            if (result->is_stop()) {
                received_count++;
            }
            // a finished completion that the server answers with a retryable error (an empty reply after thinking); the
            // task is already over, so there is nothing to cancel
            if (auto retry = result->to_retryable_error()) {
                return retry;
            }
            return result;
        }
    }

    // should not reach here
}

server_response_reader::batch_response server_response_reader::wait_for_all(const std::function<bool()> & should_stop) {
    batch_response batch_res;
    batch_res.results.clear();
    batch_res.results.resize(id_tasks.size());
    while (has_next()) {
        auto res = next(should_stop);
        if (res == nullptr) {
            batch_res.is_terminated = true;
            return batch_res;
        }
        if (res->is_error()) {
            batch_res.error = std::move(res);
            return batch_res;
        }
        const size_t idx = res->index;
        GGML_ASSERT(idx < batch_res.results.size() && "index out of range");
        GGML_ASSERT(batch_res.results[idx] == nullptr && "duplicate result received");
        batch_res.results[idx] = std::move(res);
    }
    return batch_res;
}

void server_response_reader::stop() {
    queue_results.remove_waiting_task_ids(id_tasks);
    if (has_next() && !cancelled) {
        // if tasks is not finished yet, cancel them
        cancelled = true;
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            SRV_WRN("cancel task, id_task = %d\n", id_task);
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            queue_results.remove_waiting_task_id(id_task);
            cancel_tasks.push_back(std::move(task));
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(std::move(cancel_tasks), true);
    } else {
        SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
    }
}

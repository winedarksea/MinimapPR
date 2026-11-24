// Bench override: production lwipopts plus compile-time knobs under test.
#pragma once
#include "../sirith_tetrahedral/lwipopts.h"

// lwIP's RTO floor is 3 slow-timer ticks (sv decays to 3, see tcp_in.c RTT
// update), i.e. 1.5 s at the default 250 ms TCP_TMR_INTERVAL even on a 3 ms
// RTT LAN. A finer timer lowers that floor proportionally.
#ifdef BENCH_TCP_TMR_INTERVAL_MS
#undef TCP_TMR_INTERVAL
#define TCP_TMR_INTERVAL BENCH_TCP_TMR_INTERVAL_MS
#endif

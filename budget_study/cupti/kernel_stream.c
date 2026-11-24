/* CUPTI injection library for the online controller: streams per-kernel
 * durations of one CUDA process.
 *
 * Loaded through CUDA_INJECTION64_PATH=<this .so>; writes nothing unless
 * KTRACE_DIR is set. Every 300 ms (the controller's step) the activity
 * buffers are flushed and the kernels completed since the last flush are
 * appended to $KTRACE_DIR/<pid>.stream, summed per kernel name and 50 ms
 * wall-clock bin (CLOCK_REALTIME, the same clock as Python's time.time()):
 *   N,<id>,<name>                       first time a kernel name is seen
 *   K,<bin start ms>,<id>,<count>,<sum ns>
 * The controller tags each bin with the clock it ran at (kshield.py).
 * kernel_trace.c is the offline version used by kernel_sweep.py.
 */
#include <cupti.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BUF_SIZE (8 * 1024 * 1024)
#define NBUCKET 65536
#define BIN_NS 50000000ULL

typedef struct Name { char *s; int id; struct Name *next; } Name;
typedef struct Bin { uint64_t bin; int id; uint64_t n, sum; struct Bin *next; } Bin;
static Name *names[NBUCKET];
static Bin *bins[NBUCKET];
static int next_id = 0;
static int64_t offset_ns = 0;   /* CLOCK_REALTIME - CUPTI timestamp */
static FILE *out = NULL;
static uint64_t dropped = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static volatile int running = 1;
static pthread_t flusher;

static uint64_t hash(const char *s) { uint64_t h = 1469598103934665603ULL; while (*s) h = (h ^ (unsigned char)*s++) * 1099511628211ULL; return h; }

static void sync_clock(void) {
  struct timespec ts; uint64_t c = 0;
  clock_gettime(CLOCK_REALTIME, &ts);
  if (cuptiGetTimestamp(&c) == CUPTI_SUCCESS)
    offset_ns = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec - (int64_t)c;
}

static int name_id(const char *s) {
  uint64_t b = hash(s) % NBUCKET;
  Name *e = names[b];
  while (e && strcmp(e->s, s)) e = e->next;
  if (e) return e->id;
  e = calloc(1, sizeof *e); e->s = strdup(s); e->id = next_id++; e->next = names[b]; names[b] = e;
  if (out) fprintf(out, "N,%d,%s\n", e->id, s);
  return e->id;
}

static void add(const char *name, uint64_t start, uint64_t end) {
  int id = name_id(name);
  uint64_t bin = (uint64_t)((int64_t)start + offset_ns) / BIN_NS;
  uint64_t b = (bin * 1315423911ULL + (uint64_t)id) % NBUCKET;
  Bin *e = bins[b];
  while (e && (e->bin != bin || e->id != id)) e = e->next;
  if (!e) { e = calloc(1, sizeof *e); e->bin = bin; e->id = id; e->next = bins[b]; bins[b] = e; }
  e->n++; e->sum += end > start ? end - start : 0;
}

static void write_bins(void) {
  for (int i = 0; i < NBUCKET; i++) {
    Bin *e = bins[i];
    while (e) {
      Bin *nx = e->next;
      if (out) fprintf(out, "K,%lu,%d,%lu,%lu\n", (unsigned long)(e->bin * (BIN_NS / 1000000ULL)), e->id, e->n, e->sum);
      free(e); e = nx;
    }
    bins[i] = NULL;
  }
  if (out) fflush(out);
}

static void CUPTIAPI buf_requested(uint8_t **buf, size_t *size, size_t *maxrec) {
  *buf = aligned_alloc(8, BUF_SIZE); *size = BUF_SIZE; *maxrec = 0;
}

static void CUPTIAPI buf_completed(CUcontext c, uint32_t s, uint8_t *buf, size_t size, size_t valid) {
  CUpti_Activity *r = NULL; size_t d = 0;
  pthread_mutex_lock(&mu);
  while (cuptiActivityGetNextRecord(buf, valid, &r) == CUPTI_SUCCESS)
    if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)r;
      add(k->name ? k->name : "?", k->start, k->end);
    }
  cuptiActivityGetNumDroppedRecords(c, s, &d); dropped += d;
  pthread_mutex_unlock(&mu);
  free(buf);
}

static void flush_now(int flag) {
  cuptiActivityFlushAll(flag);
  pthread_mutex_lock(&mu);
  write_bins();
  sync_clock();
  pthread_mutex_unlock(&mu);
}

static void *flush_loop(void *arg) {
  struct timespec ts = {0, 300000000L};
  while (running) { nanosleep(&ts, NULL); flush_now(0); }
  return NULL;
}

static void at_exit(void) {
  running = 0;
  flush_now(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  if (dropped) fprintf(stderr, "kernel_stream: %lu records dropped\n", dropped);
  if (out) fclose(out);
  out = NULL;
}

int InitializeInjection(void) {
  const char *dir = getenv("KTRACE_DIR");
  if (!dir) return 1;
  char p[4096];
  snprintf(p, sizeof p, "%s/%d.stream", dir, getpid());
  if (!(out = fopen(p, "w"))) return 0;
  if (cuptiActivityRegisterCallbacks(buf_requested, buf_completed) != CUPTI_SUCCESS) return 0;
  if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS) return 0;
  sync_clock();
  atexit(at_exit);
  pthread_create(&flusher, NULL, flush_loop, NULL);
  return 1;
}

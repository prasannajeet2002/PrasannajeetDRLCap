/* CUPTI injection library: per-kernel durations of one CUDA process.
 *
 * Loaded by the CUDA driver through CUDA_INJECTION64_PATH=<this .so>.
 * Records every kernel (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), flushes the
 * activity buffers every 300 ms (the controller's step, so the overhead
 * measured here is what an online controller would pay) and at exit writes
 *   $KTRACE_DIR/<pid>_kernels.csv   name,count,sum_ns,min_ns,max_ns
 *   $KTRACE_DIR/<pid>_windows.csv   window,busy_ns,kernels   (300 ms windows)
 * Nothing is written if KTRACE_DIR is unset.
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
#define WIN_NS 300000000ULL
#define MAXWIN 1000000

typedef struct Ent { char *name; uint64_t n, sum, mn, mx; struct Ent *next; } Ent;
static Ent *tab[NBUCKET];
static uint64_t *win_busy, *win_n, t0 = 0, nwin = 0, dropped = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static volatile int running = 1;
static pthread_t flusher;

static uint64_t hash(const char *s) { uint64_t h = 1469598103934665603ULL; while (*s) h = (h ^ (unsigned char)*s++) * 1099511628211ULL; return h; }

static void add(const char *name, uint64_t start, uint64_t end) {
  uint64_t d = end > start ? end - start : 0, b = hash(name) % NBUCKET;
  Ent *e = tab[b];
  while (e && strcmp(e->name, name)) e = e->next;
  if (!e) { e = calloc(1, sizeof *e); e->name = strdup(name); e->mn = UINT64_MAX; e->next = tab[b]; tab[b] = e; }
  e->n++; e->sum += d; if (d < e->mn) e->mn = d; if (d > e->mx) e->mx = d;
  if (!t0) t0 = start;
  if (start >= t0) { uint64_t w = (start - t0) / WIN_NS; if (w < MAXWIN) { win_busy[w] += d; win_n[w]++; if (w + 1 > nwin) nwin = w + 1; } }
}

static void CUPTIAPI buf_requested(uint8_t **buf, size_t *size, size_t *maxrec) {
  *buf = aligned_alloc(8, BUF_SIZE); *size = BUF_SIZE; *maxrec = 0;
}

static void CUPTIAPI buf_completed(CUcontext c, uint32_t s, uint8_t *buf, size_t size, size_t valid) {
  CUpti_Activity *r = NULL; size_t d = 0;
  pthread_mutex_lock(&mu);
  while (cuptiActivityGetNextRecord(buf, valid, &r) == CUPTI_SUCCESS) {
    if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)r;
      add(k->name ? k->name : "?", k->start, k->end);
    }
  }
  cuptiActivityGetNumDroppedRecords(c, s, &d); dropped += d;
  pthread_mutex_unlock(&mu);
  free(buf);
}

static void *flush_loop(void *arg) {
  struct timespec ts = {0, 300000000L};
  while (running) { nanosleep(&ts, NULL); cuptiActivityFlushAll(0); }
  return NULL;
}

static void at_exit(void) {
  running = 0;
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  const char *dir = getenv("KTRACE_DIR");
  if (!dir) return;
  char p[4096]; FILE *f;
  pthread_mutex_lock(&mu);
  snprintf(p, sizeof p, "%s/%d_kernels.csv", dir, getpid());
  if ((f = fopen(p, "w"))) {
    fprintf(f, "name,count,sum_ns,min_ns,max_ns\n");
    for (int i = 0; i < NBUCKET; i++)
      for (Ent *e = tab[i]; e; e = e->next) {
        fputc('"', f); for (char *q = e->name; *q; q++) { if (*q == '"') fputc('"', f); fputc(*q, f); }
        fprintf(f, "\",%lu,%lu,%lu,%lu\n", e->n, e->sum, e->mn, e->mx);
      }
    fclose(f);
  }
  snprintf(p, sizeof p, "%s/%d_windows.csv", dir, getpid());
  if (nwin && (f = fopen(p, "w"))) {
    fprintf(f, "window,busy_ns,kernels\n");
    for (uint64_t w = 0; w < nwin; w++) fprintf(f, "%lu,%lu,%lu\n", w, win_busy[w], win_n[w]);
    fclose(f);
  }
  if (dropped) fprintf(stderr, "kernel_trace: %lu records dropped\n", dropped);
  pthread_mutex_unlock(&mu);
}

int InitializeInjection(void) {
  win_busy = calloc(MAXWIN, sizeof *win_busy); win_n = calloc(MAXWIN, sizeof *win_n);
  if (cuptiActivityRegisterCallbacks(buf_requested, buf_completed) != CUPTI_SUCCESS) return 0;
  if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS) return 0;
  atexit(at_exit);
  pthread_create(&flusher, NULL, flush_loop, NULL);
  return 1;
}

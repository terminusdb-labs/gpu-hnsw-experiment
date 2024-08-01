#include <stdint.h>
#include <cuda_bf16.h>

struct ring_queue_t {
  long *ids; 
  __nv_bfloat16 *priorities;
  size_t head;
  size_t tail;
  size_t capacity;
};

struct pair_t {
  long *id;
  __nv_bfloat16 *priority;
};

namespace ring { 
  __device__ size_t ring_relative_to_absolute(ring_queue_t *rq, size_t relative_index) {
    if (relative_index < (rq->capacity - rq->head)) {
      return rq->head + relative_index;
    }else{
      return relative_index - (rq->capacity - rq->head);
    }
  }

  __device__ size_t ring_absolute_to_relative(ring_queue_t *rq, size_t absolute_index) {
    if (absolute_index < rq->head) {
      return absolute_index + (rq->capacity - rq->head);
    }else{
      return absolute_index - rq->head;
    }
  }

  __device__ bool ring_is_empty(ring_queue_t *rq) {
    if (rq->head == rq->tail) {
      return true;
    } else {
      return false;
    }
  }

  __device__ pair_t ring_first(ring_queue_t *rq) {
    struct pair_t pair;
    if (ring_is_empty(rq)) {
      pair.id = nullptr;
      pair.priority = nullptr;
    } else {
      pair.id = &(rq->ids[rq->head]);
      pair.priority = &(rq->priorities[rq->head]);
    }
    return pair;
  }

  __device__ int ring_advance(ring_queue_t * rq) {
    if (ring_is_empty(rq)) {
      return -1;
    } else {
      if (rq->head == rq->capacity-1) {
	rq->head = 0;
	return 0;
      }else{
	rq->head++;
	return 0;
      }
    }
  }

  __device__ pair_t ring_pop_first(ring_queue_t *rq) {
    struct pair_t pair;
    if (ring_is_empty(rq)) {
      pair.id = nullptr;
      pair.priority = nullptr;
    } else {
      pair.id = &(rq->ids[rq->head]);
      pair.priority = &(rq->priorities[rq->head]);
      ring_advance(rq);
    }
    return pair;
  }

  __device__ size_t ring_len(ring_queue_t *rq) {
    if (rq->tail > rq->head) {
      return (rq->tail - rq->head);
    } else {
      return ((rq->capacity - rq->head) + rq->tail);
    }
  }

  __device__ pair_t ring_get(ring_queue_t *rq, size_t idx) {
    struct pair_t pair;
    if (idx < ring_len(rq)) {
      size_t aidx = ring_relative_to_absolute(rq, idx);
      pair.id = &(rq->ids[aidx]);
      pair.priority = &(rq->priorities[aidx]);
    } else {
      pair.id = nullptr;
      pair.priority = nullptr;
    }
    return pair;
  }

  __device__ int ring_set(ring_queue_t *rq, size_t idx, long id, __nv_bfloat16 priority) {
    size_t aidx = ring_relative_to_absolute(rq, idx);
    rq->ids[aidx] = id;
    rq->priorities[aidx] = priority;
    return 0;
  }

}

extern "C" __global__ size_t ring_relative_to_absolute(ring_queue_t *rq, size_t relative_index) {
  return ring::ring_relative_to_absolute(rq, relative_index);
}

extern "C" __global__ size_t ring_absolute_to_relative(ring_queue_t *rq, size_t absolute_index) {
  return ring::ring_absolute_to_relative(rq, absolute_index);
}

extern "C" __global__ bool ring_is_empty(ring_queue_t *rq) {
  return ring::ring_is_empty(rq);
}

extern "C" __global__ pair_t ring_first(ring_queue_t *rq) {
  return ring::ring_first(rq);
}

extern "C" __global__ int ring_advance(ring_queue_t *rq) {
  return ring::ring_advance(rq);
}

extern "C" __global__ pair_t ring_pop_first(ring_queue_t *rq) {
  return ring::ring_pop_first(rq);
}

extern "C" __global__ size_t ring_len(ring_queue_t *rq) {
  return ring::ring_len(rq);
}

extern "C" __global__ pair_t ring_get(ring_queue_t *rq, size_t idx) {
  return ring::ring_get(rq, idx);
}

extern "C" __global__ int ring_set(ring_queue_t *rq, size_t idx, long id, __nv_bfloat16 priority) {
  return ring::ring_set(rq, idx, id, priority);
}

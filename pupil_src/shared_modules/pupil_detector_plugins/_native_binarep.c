#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define BINAREP_CHANNELS 2
#define BINAREP_HEIGHT 60
#define BINAREP_WIDTH 80
#define BINAREP_TIME_BINS 4
#define BINAREP_SIZE (BINAREP_CHANNELS * BINAREP_HEIGHT * BINAREP_WIDTH)

/*
 * Build:
 *   cc -O3 -shared -fPIC _native_binarep.c -o _native_binarep.so
 *
 * Returns 0 on success and a negative value for invalid arguments.
 */
int pupil_fill_binarep(
    const int64_t *xs,
    const int64_t *ys,
    const int64_t *timestamps_us,
    const uint8_t *polarities,
    size_t event_count,
    int64_t slice_start_us,
    int64_t time_window_us,
    uint8_t *output
) {
    size_t index;

    if (output == NULL || time_window_us <= 0) {
        return -1;
    }
    memset(output, 0, BINAREP_SIZE * sizeof(uint8_t));
    if (event_count == 0) {
        return 0;
    }
    if (xs == NULL || ys == NULL || timestamps_us == NULL || polarities == NULL) {
        return -2;
    }

    for (index = 0; index < event_count; ++index) {
        int64_t x = xs[index];
        int64_t y = ys[index];
        int64_t relative_t;
        int64_t bin;
        size_t output_index;

        if (x < 0 || x >= BINAREP_WIDTH || y < 0 || y >= BINAREP_HEIGHT) {
            continue;
        }
        relative_t = timestamps_us[index] - slice_start_us;
        if (relative_t < 0) {
            relative_t = 0;
        } else if (relative_t >= time_window_us) {
            relative_t = time_window_us - 1;
        }
        bin = (relative_t * BINAREP_TIME_BINS) / time_window_us;
        output_index =
            ((polarities[index] != 0) ? 1U : 0U) * BINAREP_HEIGHT * BINAREP_WIDTH
            + (size_t)y * BINAREP_WIDTH
            + (size_t)x;
        output[output_index] |= (uint8_t)(1U << bin);
    }
    return 0;
}

/*
 * Exact counterpart of the original Python/tonic preprocessing:
 *   ToFrame(n_time_bins=4) -> ToBinaRep(n_frames=1, n_bits=4)
 *
 * Tonic builds equal-width bins from the packet's first and last timestamps,
 * ignores a trailing remainder, weights the earliest bin as bit 3, and later
 * normalises the uint8 result by 15 before it reaches the network.
 */
int pupil_fill_binarep_legacy(
    const int64_t *xs,
    const int64_t *ys,
    const int64_t *timestamps_us,
    const uint8_t *polarities,
    size_t event_count,
    uint8_t *output
) {
    size_t index;
    int64_t first_timestamp;
    int64_t duration_us;
    int64_t bin_width_us;

    if (output == NULL) {
        return -1;
    }
    memset(output, 0, BINAREP_SIZE * sizeof(uint8_t));
    if (event_count == 0) {
        return 0;
    }
    if (xs == NULL || ys == NULL || timestamps_us == NULL || polarities == NULL) {
        return -2;
    }

    first_timestamp = timestamps_us[0];
    duration_us = timestamps_us[event_count - 1] - first_timestamp;
    bin_width_us = duration_us / BINAREP_TIME_BINS;
    /* SliceByTimeBins yields empty frames when the duration is less than 4 us. */
    if (bin_width_us <= 0) {
        return 0;
    }

    for (index = 0; index < event_count; ++index) {
        int64_t x = xs[index];
        int64_t y = ys[index];
        int64_t relative_t;
        int64_t bin;
        size_t output_index;

        if (x < 0 || x >= BINAREP_WIDTH || y < 0 || y >= BINAREP_HEIGHT) {
            continue;
        }
        relative_t = timestamps_us[index] - first_timestamp;
        if (relative_t < 0) {
            continue;
        }
        bin = relative_t / bin_width_us;
        /* Tonic's searchsorted end bounds exclude the trailing remainder. */
        if (bin < 0 || bin >= BINAREP_TIME_BINS) {
            continue;
        }
        output_index =
            ((polarities[index] != 0) ? 1U : 0U) * BINAREP_HEIGHT * BINAREP_WIDTH
            + (size_t)y * BINAREP_WIDTH
            + (size_t)x;
        output[output_index] |= (uint8_t)(1U << (BINAREP_TIME_BINS - 1 - bin));
    }
    return 0;
}

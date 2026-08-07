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


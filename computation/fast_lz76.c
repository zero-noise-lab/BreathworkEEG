/* Adapted from NeuroKit2 0.2.12 (MIT), copyright (c) 2020 Dominique Makowski.
 * https://github.com/neuropsychology/NeuroKit/blob/0.2.12/neurokit2/complexity/complexity_lempelziv.py
 * Upstream MIT notice: ../LICENSE. References: ../README.md.
 */
#include <stddef.h>
#include <stdint.h>

/*
 * Exact compiled equivalent of NeuroKit2 0.2.12's
 * _complexity_lempelziv_count / Kaspar-Schuster-style LZ76 parser.
 *
 * The Python implementation is easy to read but becomes impractical when
 * 63 channels x 2500 samples are concatenated.  This implementation changes
 * only execution speed; phrase boundaries and counts are identical.
 */
uint64_t lz76_count(const uint8_t *symbols, size_t n) {
    if (symbols == NULL || n < 2) {
        return 0;
    }

    size_t complexity = 1;
    size_t lookahead_start = 1;
    size_t search_index = 0;
    size_t word_length = 1;
    size_t longest_match = 1;

    for (;;) {
        /* Python prepends one sentinel and then accesses s[index + length].
         * As both sums are >=1, subtracting one indexes the input symbols. */
        const uint8_t search_symbol = symbols[search_index + word_length - 1];
        const uint8_t lookahead_symbol = symbols[lookahead_start + word_length - 1];

        if (search_symbol != lookahead_symbol) {
            if (word_length > longest_match) {
                longest_match = word_length;
            }
            search_index += 1;

            if (search_index == lookahead_start) {
                complexity += 1;
                lookahead_start += longest_match;
                if (lookahead_start + 1 > n) {
                    break;
                }
                search_index = 0;
                word_length = 1;
                longest_match = 1;
            } else {
                word_length = 1;
            }
        } else {
            word_length += 1;
            if (lookahead_start + word_length > n) {
                complexity += 1;
                break;
            }
        }
    }
    return (uint64_t)complexity;
}

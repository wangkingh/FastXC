#include "cal_nseg.h"

// The function finds the smallest number greater than the input 'm'
// that can be factored by 2, 3, 5, and 7.
int findOptimalTransformLength(int inputLength)
{
    int optimalLength = inputLength;
    int i;
    while (1)
    {
        int remainingFactor = optimalLength;
        int primeFactors[] = {2, 3, 5, 7}; // The prime factors used by CUFFT
        for (i = 0; i < 4; i++)
        {
            // Continuously divide by each factor if possible
            while ((remainingFactor > 1) &&
                   (remainingFactor % primeFactors[i] == 0))
            {
                remainingFactor = remainingFactor / primeFactors[i];
            }
        }
        // If all factors have been divided out, then we have found our answer
        if (remainingFactor == 1)
        {
            break;
        }
        optimalLength = optimalLength + 1;
    }
    return optimalLength;
}

/*
 * findLargestTransformLengthBelow - Find the largest number smaller than
 * 'inputLength' that can be factored by 2, 3, 5, and 7.
 */
int findLargestTransformLengthBelow(int inputLength, int *error)
{
    // If the input length is less than or equal to 1, it's invalid
    if (inputLength <= 1)
    {
        *error = EINVAL; // Set the error code to 'EINVAL' (Invalid Argument)
        return 0;        // Return 0 as an error occurred
    }

    int optimalLength = inputLength - 1; // Start from the number less than input
    int i;
    // Keep reducing 'optimalLength' until we find a suitable number or reach 0
    while (optimalLength > 0)
    {
        int remainingFactor = optimalLength;
        // The prime factors used
        int primeFactors[] = {2, 3, 5, 7};
        for (i = 0; i < 4; i++)
        {
            // Try to divide 'remainingFactor' by each prime factor as long as
            // possible
            while ((remainingFactor > 1) &&
                   (remainingFactor % primeFactors[i] == 0))
            {
                remainingFactor = remainingFactor / primeFactors[i];
            }
        }

        // If 'remainingFactor' is 1, 'optimalLength' can be factored by 2, 3, 5,
        // and 7
        if (remainingFactor == 1)
        {
            *error = 0;           // Set the error code to 0 indicating success
            return optimalLength; // Return the suitable number found
        }

        optimalLength = optimalLength - 1; // Decrease 'optimalLength'
    }

    *error = EINVAL; // If we reach here, no suitable number was found. Set the
                     // error code to 'EINVAL'
    return 0;        // Return 0 as an error occurred
}

int cal_nseg(int seglen, int npts, float delta)
{
    int seg_npts = seglen / delta;
    int nfft = findOptimalTransformLength(seg_npts);

    if (nfft > npts)
    {
        nfft = npts;
        int err;
        nfft = findLargestTransformLengthBelow(nfft, &err);
        if (err)
        {
            fprintf(stderr, "Error finding optimal of transform length\n");
        }
        else
        {
            printf("nfft is set to %d\n", nfft);
        }
    }
    int nseg = nfft;
    return nseg;
}
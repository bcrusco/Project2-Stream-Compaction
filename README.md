CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Bradley Crusco
* Tested on: Windows 10, i7-3770K @ 3.50GHz 16GB, 2 x GTX 980 4096MB (Personal Computer)

![](images/Project 2 Analysis.png "Performance Analysis")
Unfortunately, the results from testing are not very impressive. The sequential CPU implementation easily out performs everything but the Thrust implementation, and the worst performer by far is the work-efficient implementation, which we'd expect to outperform the naive scan. So why is this? I am not 100% sure. However I had difficulty determining how to configure the grid and block size optimally, and as a result all the GPU implementations are using the same ratio, with 512 threads per block. A better understanding of how to configure this might result in performance more in line with what we'd expect to see.

The other possible cause may be that our arrays are not very large, with the maximum array I tested with being 1024. It could be the case that this wasn't enough data for the GPU to take advantage of and counteract the overhead of the parallel algorithm vs. the sequential.

```
****************
** SCAN TESTS **
****************
    [   3  29  33  19   0  16  10  40  39  50  44  30   9 ...   4   0 ]
==== cpu scan, power-of-two ====
CPU execution time for scan: 0.00109ms
    [   0   3  32  65  84  84 100 110 150 189 239 283 313 ... 6684 6688 ]
==== cpu scan, non-power-of-two ====
CPU execution time for scan: 0.00106ms
    [   0   3  32  65  84  84 100 110 150 189 239 283 313 ... 6613 6626 ]
    passed
==== naive scan, power-of-two ====
CUDA execution time for naive scan: 0.07440ms
    passed
==== naive scan, non-power-of-two ====
CUDA execution time for naive scan: 0.07222ms
    passed
==== work-efficient scan, power-of-two ====
CUDA execution time for work efficient scan: 0.21798ms
    passed
==== work-efficient scan, non-power-of-two ====
CUDA execution time for work efficient scan: 0.21632ms
    passed
==== thrust scan, power-of-two ====
    passed
==== thrust scan, non-power-of-two ====
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   4   3   0   3   4   2   3   2   3   1   1   1   4 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
CPU execution time for compact without scan: 0.00106ms
    [   4   3   3   4   2   3   2   3   1   1   1   4   3 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
CPU execution time for compact without scan: 0.00106ms
    [   4   3   3   4   2   3   2   3   1   1   1   4   3 ...   4   4 ]
    passed
==== cpu compact with scan ====
CPU execution time for compact with scan: 0.00109ms
    [   4   3   3   4   2   3   2   3   1   1   1   4   3 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
CUDA execution time for stream compaction: 0.22755ms
    passed
==== work-efficient compact, non-power-of-two ====
CUDA execution time for stream compaction: 0.22557ms
    passed
```

## Write-up

1. Update all of the TODOs at the top of this README.
2. Add a description of this project including a list of its features.
3. Add your performance analysis (see below).

All extra credit features must be documented in your README, explaining its
value (with performance comparison, if applicable!) and showing an example how
it works. For radix sort, show how it is called and an example of its output.

Always profile with Release mode builds and run without debugging.

### Questions

* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.
  * (You shouldn't compare unoptimized implementations to each other!)

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * You should use CUDA events for timing. Be sure **not** to include any
    explicit memory operations in your performance measurements, for
    comparability.
  * To guess at what might be happening inside the Thrust implementation, take
    a look at the Nsight timeline for its execution.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

These questions should help guide you in performance analysis on future
assignments, as well.

## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project 2: PENNKEY`
   * Direct link to your pull request on GitHub
   * In the form of a grade (0-100+) with comments, evaluate your own
     performance on the project.
   * Feedback on the project itself, if any.

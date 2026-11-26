#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "ModelicaUtilities.h"

void weightedSum(const double *a, size_t n, size_t m, const double *w, double *y) {
    for (size_t i = 0; i < n; i++) {
        y[i] = 0;
        for (size_t j = 0; j < m; j++) {
            y[i] += a[i * m + j] * w[j];
        }
    }
}

void scaleShift(double x, int k, int negate, int c, double *y, int *m) {
    *y = (negate ? -x : x) * k;
    *m = k + c;
}

const char *describe(const char *name, double x) {
    char *s = ModelicaAllocateString(63);
    snprintf(s, 64, "%s=%g", name, x);
    return s;
}

double checkedSqrt(double x) {
    if (x < 0) {
        ModelicaFormatError("checkedSqrt: negative input %g", x);
    }
    return sqrt(x);
}

double sumWithWork(const double *x, size_t nx, double *work, int n) {
    double sum = 0;
    for (size_t i = 0; i < nx; i++) {
        sum += x[i];
    }
    for (int j = 0; j < n; j++) {
        sum += work[j];
    }
    return sum;
}

double lookupScale(const char *key, double x) {
    return strcmp(key, "double") == 0 ? 2 * x : x;
}

double cube(double x, int k) {
    return k * x * x * x;
}

double cubeDer(double x, int k, double der_x) {
    return 3 * k * x * x * der_x;
}

double cubeDer2(double x, int k, double der_x, double der2_x) {
    return 6 * k * x * der_x * der_x + 3 * k * x * x * der2_x;
}

void addInPlace(double x, double *y, double *w, size_t n) {
    *y += x;
    for (size_t i = 0; i < n; i++) {
        w[i] += 1;
    }
}

double addTwo(double a, double b) {
    return a + b;
}

double dotTable(const double *x, const double *table, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * table[i];
    }
    return sum;
}

/* Column-major: element (i, j) is at i + j * n */
void addIndices(const double *a, size_t n, size_t m, double *b) {
    for (size_t j = 0; j < m; j++) {
        for (size_t i = 0; i < n; i++) {
            b[i + j * n] = a[i + j * n] + 10 * i + j;
        }
    }
}

void lengths(const char **names, size_t n, double *y) {
    for (size_t i = 0; i < n; i++) {
        y[i] = (double)strlen(names[i]);
    }
}

double square(double x) {
    return x * x;
}

double scaled(double x, double k) {
    return k * x;
}

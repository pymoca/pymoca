#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "ModelicaUtilities.h"

typedef struct {
    double *array;
    size_t size;
} MyTable;

static int liveTables = 0;

int countMyTables(void) {
    return liveTables;
}

void *initMyTable(const double *vals, size_t size) {
    MyTable *table = malloc(sizeof(MyTable));
    table->array = malloc(size * sizeof(double));
    memcpy(table->array, vals, size * sizeof(double));
    table->size = size;
    liveTables++;
    return table;
}

void closeMyTable(void *object) {
    MyTable *table = object;
    free(table->array);
    free(table);
    liveTables--;
}

double interpolateMyTable(void *object, double u) {
    MyTable *table = object;
    if (u < 0 || u > table->size - 1) {
        ModelicaFormatError("interpolateMyTable: %g is outside of the table", u);
    }
    size_t i = (size_t)u < table->size - 1 ? (size_t)u : table->size - 2;
    return table->array[i] + (u - i) * (table->array[i + 1] - table->array[i]);
}

double interpolateMyTableDer(void *object, double u, double der_u) {
    MyTable *table = object;
    size_t i = (size_t)u < table->size - 1 ? (size_t)u : table->size - 2;
    return (table->array[i + 1] - table->array[i]) * der_u;
}

int stringLength(const char *s) {
    return (int)strlen(s);
}

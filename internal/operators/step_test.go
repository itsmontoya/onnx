package operators

import (
	"reflect"
	"testing"
)

func TestStepFixedMetadataReservation(t *testing.T) {
	// Count scalar payload recursively, excluding shared strings, pointers and
	// separately charged slices. This includes both CNN configs and version flags.
	bytes := stepNumericBytes(reflect.TypeFor[Step]())
	if bytes > fixedStepBytes {
		t.Fatalf("step numeric metadata %d exceeds reservation %d", bytes, fixedStepBytes)
	}
}

func stepNumericBytes(value reflect.Type) (bytes int64) {
	if value.Kind() == reflect.Struct {
		for index := 0; index < value.NumField(); index++ {
			bytes += stepNumericBytes(value.Field(index).Type)
		}
		return bytes
	}

	if value.Kind() == reflect.Array {
		return int64(value.Len()) * stepNumericBytes(value.Elem())
	}

	switch value.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int64, reflect.Float32:
		return int64(value.Size())
	}
	return 0
}

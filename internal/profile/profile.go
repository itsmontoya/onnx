// Package profile defines the exact model envelopes audited by the runtime.
package profile

// Accepts reports the two tested standard-domain profiles. Gaps are unsupported.
func Accepts(ir int64, domain string, opset int64) (accepted bool) {
	return ir == 8 && domain == "" && (opset == 13 || opset == 17)
}

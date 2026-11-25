package onnx

// SessionGPUStats reports Metal execution counters from the most recent
// completed or attempted Run on a Session, along with cached activation storage.
type SessionGPUStats struct {
	Active            bool
	DeviceOps         int
	HostFallbackOps   int
	ScopeSubmits      int
	HostUploads       int
	HostDownloads     int
	ActivationBuffers int
	ActivationBytes   uint64
}

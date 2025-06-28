class ContextBuffer:
    def __init__(self, N=3):
        self.N = N
        self.buffer = []

    def add(self, situation):
        """Add new situations to the buffer, delete the oldest ones if full."""
        self.buffer.append(situation)
        if len(self.buffer) > self.N:
            self.buffer.pop(0)

    def get_context_string(self, sep=" [CTX] "):
        """Combine buffers into a single context string (for API)."""
        return sep.join(self.buffer)

    def get_context_list(self):
        """Return history list (for API)."""
        return list(self.buffer)

    def clear(self):
        self.buffer = []

# ====== DEMO ======
if __name__ == "__main__":
    cb = ContextBuffer(N=3)
    situations = [
        "Enemy is 1.1 meters ahead angle 90 degrees dash ready",
        "Enemy is 1.0 meters ahead angle 85 degrees dash ready",
        "Enemy is 0.8 meters ahead angle 80 degrees dash ready",
        "Enemy is 0.7 meters ahead angle 75 degrees dash ready"
    ]
    for s in situations:
        cb.add(s)
        print("Buffer:", cb.buffer)
        print("Context string:", cb.get_context_string())
        print("Context list:", cb.get_context_list())

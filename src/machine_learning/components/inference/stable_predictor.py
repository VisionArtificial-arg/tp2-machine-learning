from collections import deque, Counter

class StablePredictor:
    def __init__(self, window=10):
        self.buffer = deque(maxlen=window)

    def update(self, label):
        if label is not None:
            self.buffer.append(label)

        if len(self.buffer) == 0:
            return None

        return Counter(self.buffer).most_common(1)[0][0]
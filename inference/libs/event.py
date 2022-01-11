from collections import deque
from .state import State


class EventManager:
    def __init__(self, buff_size=30):
        self.buff_size = buff_size
        self.queue = deque(maxlen=buff_size)

    def append(self, state):
        self.queue.append(state)

    def is_blinking_event(self):
        BLINKING_THRESHOLD = self.buff_size / 3
        num_red_light_on = len([s for s in self.queue if s is State.RED_LIGHT_ON])
        rest = len(self.queue) - num_red_light_on
        return (
            abs(num_red_light_on - rest) <= BLINKING_THRESHOLD
            and self.queue[-1] == self.queue[0]
        )

    def get_event_state(self):
        if self.is_blinking_event():
            return State.BLINKING
        return self.queue[-1]


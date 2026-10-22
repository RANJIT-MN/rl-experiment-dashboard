import queue
import threading


STREAM_QUEUE_SIZE = 100
MAX_SUBSCRIBERS_PER_EXPERIMENT = 50


class ExperimentMetricStream:
    def __init__(
        self,
        queue_size: int = STREAM_QUEUE_SIZE,
        max_subscribers_per_experiment: int = MAX_SUBSCRIBERS_PER_EXPERIMENT,
    ):
        self.queue_size = queue_size
        self.max_subscribers_per_experiment = max_subscribers_per_experiment
        self._subscribers: dict[int, set[queue.Queue]] = {}
        self._lock = threading.Lock()

    def subscribe(self, experiment_id: int) -> queue.Queue:
        with self._lock:
            subscribers = self._subscribers.setdefault(experiment_id, set())
            if len(subscribers) >= self.max_subscribers_per_experiment:
                if not subscribers:
                    self._subscribers.pop(experiment_id, None)
                raise RuntimeError("Too many live metric subscribers for this experiment")
            subscriber = queue.Queue(maxsize=self.queue_size)
            subscribers.add(subscriber)
            return subscriber

    def unsubscribe(self, experiment_id: int, subscriber: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(experiment_id)
            if subscribers is None:
                return
            subscribers.discard(subscriber)
            if not subscribers:
                self._subscribers.pop(experiment_id, None)

    def publish(self, experiment_id: int, event: dict) -> None:
        with self._lock:
            for subscriber in self._subscribers.get(experiment_id, ()):
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    while True:
                        try:
                            subscriber.get_nowait()
                        except queue.Empty:
                            break
                    subscriber.put_nowait({"type": "resync_required"})

    def subscriber_count(self, experiment_id: int) -> int:
        with self._lock:
            return len(self._subscribers.get(experiment_id, ()))


metric_stream = ExperimentMetricStream()


def publish_status(
    experiment_id: int,
    status: str,
    total_timesteps: int | None = None,
) -> None:
    event = {
        "type": "status",
        "experiment_id": experiment_id,
        "status": status,
    }
    if total_timesteps is not None:
        event["total_timesteps"] = total_timesteps
    metric_stream.publish(experiment_id, event)

"""
Real-time streaming data utilities for NILM
"""
import time
import numpy as np
import pandas as pd
from typing import Iterator, Dict, Callable, Optional
from collections import deque
import threading
import queue


class DataStreamSimulator:
    """Simulate a real-time data stream from pre-loaded data"""

    def __init__(
        self,
        data: pd.DataFrame,
        sample_rate: float = 1.0,
        realtime: bool = True
    ):
        """
        Initialize stream simulator

        Args:
            data: DataFrame with power data
            sample_rate: Samples per second
            realtime: If True, simulate real-time delays
        """
        self.data = data['power'].values if isinstance(data, pd.DataFrame) else data
        self.sample_rate = sample_rate
        self.realtime = realtime
        self.current_idx = 0

    def stream(self, chunk_size: int = 1) -> Iterator[np.ndarray]:
        """
        Stream data in chunks

        Args:
            chunk_size: Number of samples per chunk

        Yields:
            Numpy array of power values
        """
        while self.current_idx < len(self.data):
            end_idx = min(self.current_idx + chunk_size, len(self.data))
            chunk = self.data[self.current_idx:end_idx]

            if self.realtime and chunk_size > 0:
                # Sleep to simulate real-time
                time.sleep(chunk_size / self.sample_rate)

            yield chunk
            self.current_idx = end_idx

    def reset(self):
        """Reset stream to beginning"""
        self.current_idx = 0


class StreamingDisaggregator:
    """
    Real-time disaggregation engine for streaming data

    Processes incoming power data and outputs appliance-level predictions
    in real-time.
    """

    def __init__(
        self,
        algorithm,
        window_size: int = 100,
        update_interval: int = 10
    ):
        """
        Initialize streaming disaggregator

        Args:
            algorithm: Trained NILM algorithm
            window_size: Size of sliding window for context
            update_interval: Update predictions every N samples
        """
        self.algorithm = algorithm
        self.window_size = window_size
        self.update_interval = update_interval

        self.buffer = deque(maxlen=window_size)
        self.sample_count = 0
        self.predictions_history = {
            app: deque(maxlen=1000)
            for app in algorithm.appliance_names
        }

    def process_sample(self, power_value: float) -> Optional[Dict[str, float]]:
        """
        Process a single power sample

        Args:
            power_value: Aggregate power in watts

        Returns:
            Dictionary of appliance predictions, or None if not ready
        """
        self.buffer.append(power_value)
        self.sample_count += 1

        # Only predict every update_interval samples
        if self.sample_count % self.update_interval != 0:
            return None

        # Need minimum samples before predicting
        if len(self.buffer) < min(10, self.window_size):
            return None

        # Get predictions
        predictions = self._predict_from_buffer()

        # Store in history
        for app, value in predictions.items():
            self.predictions_history[app].append(value)

        return predictions

    def _predict_from_buffer(self) -> Dict[str, float]:
        """Predict appliance consumption from buffered data"""
        from algorithms.combinatorial_optimization import CombinatorialOptimization
        from algorithms.fhmm import FHMM

        current_power = self.buffer[-1]

        if isinstance(self.algorithm, CombinatorialOptimization):
            # CO: use current sample
            predictions = self.algorithm._find_best_combination(current_power)

        elif isinstance(self.algorithm, FHMM):
            # FHMM: use window
            chunk = np.array(list(self.buffer)).reshape(-1, 1)
            pred = self.algorithm.disaggregate_chunk(chunk)
            predictions = {
                app: pred[app][-1] if len(pred[app]) > 0 else 0
                for app in self.algorithm.appliance_names
            }

        else:
            # Unknown algorithm, return zeros
            predictions = {app: 0 for app in self.algorithm.appliance_names}

        return predictions

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics on recent predictions"""
        stats = {}

        for app, history in self.predictions_history.items():
            if len(history) > 0:
                values = np.array(list(history))
                stats[app] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'current': values[-1]
                }

        return stats


class AsyncStreamProcessor:
    """
    Asynchronous stream processor for non-blocking disaggregation

    Processes data in a separate thread to avoid blocking the main application.
    """

    def __init__(
        self,
        algorithm,
        callback: Optional[Callable] = None,
        buffer_size: int = 1000
    ):
        """
        Initialize async processor

        Args:
            algorithm: Trained NILM algorithm
            callback: Function to call with predictions (optional)
            buffer_size: Size of input queue buffer
        """
        self.algorithm = algorithm
        self.callback = callback

        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue()

        self.disaggregator = StreamingDisaggregator(algorithm)

        self.running = False
        self.thread = None

    def start(self):
        """Start the processing thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the processing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def add_sample(self, power_value: float):
        """
        Add a power sample to process

        Args:
            power_value: Aggregate power in watts
        """
        try:
            self.input_queue.put(power_value, block=False)
        except queue.Full:
            # Drop sample if queue is full
            pass

    def get_predictions(self, block: bool = False, timeout: float = None) -> Optional[Dict]:
        """
        Get latest predictions

        Args:
            block: Wait for predictions if not available
            timeout: Timeout for blocking wait

        Returns:
            Dictionary of predictions or None
        """
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def _process_loop(self):
        """Main processing loop (runs in thread)"""
        while self.running:
            try:
                # Get sample from queue
                power_value = self.input_queue.get(timeout=0.1)

                # Process
                predictions = self.disaggregator.process_sample(power_value)

                if predictions is not None:
                    # Put in output queue
                    try:
                        self.output_queue.put(predictions, block=False)
                    except queue.Full:
                        # Drop old predictions if queue is full
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.put(predictions, block=False)
                        except queue.Empty:
                            pass

                    # Call callback if provided
                    if self.callback:
                        self.callback(predictions)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue


class StreamMetricsCollector:
    """Collect metrics on streaming disaggregation performance"""

    def __init__(self):
        """Initialize metrics collector"""
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.sample_count = 0
        self.prediction_count = 0
        self.start_time = time.time()
        self.processing_times = deque(maxlen=1000)

        self.ground_truth_buffer = {}
        self.prediction_buffer = {}

    def add_sample(
        self,
        ground_truth: Optional[Dict[str, float]] = None,
        predictions: Optional[Dict[str, float]] = None,
        processing_time: Optional[float] = None
    ):
        """
        Add a sample to metrics

        Args:
            ground_truth: Actual appliance values
            predictions: Predicted appliance values
            processing_time: Time taken to process (seconds)
        """
        self.sample_count += 1

        if predictions is not None:
            self.prediction_count += 1

        if processing_time is not None:
            self.processing_times.append(processing_time)

        if ground_truth:
            for app, value in ground_truth.items():
                if app not in self.ground_truth_buffer:
                    self.ground_truth_buffer[app] = deque(maxlen=1000)
                self.ground_truth_buffer[app].append(value)

        if predictions:
            for app, value in predictions.items():
                if app not in self.prediction_buffer:
                    self.prediction_buffer[app] = deque(maxlen=1000)
                self.prediction_buffer[app].append(value)

    def get_report(self) -> str:
        """Generate metrics report"""
        elapsed = time.time() - self.start_time

        report = []
        report.append("="*60)
        report.append("STREAMING DISAGGREGATION METRICS")
        report.append("="*60)

        report.append(f"\nSamples processed: {self.sample_count:,}")
        report.append(f"Predictions made: {self.prediction_count:,}")
        report.append(f"Elapsed time: {elapsed:.1f}s")
        report.append(f"Throughput: {self.sample_count/elapsed:.1f} samples/sec")

        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            report.append(f"Avg processing time: {avg_time:.2f}ms")

        # Calculate accuracy if we have ground truth
        if self.ground_truth_buffer and self.prediction_buffer:
            report.append("\nAccuracy Metrics:")

            for app in self.ground_truth_buffer.keys():
                if app in self.prediction_buffer:
                    gt = np.array(list(self.ground_truth_buffer[app]))
                    pred = np.array(list(self.prediction_buffer[app]))

                    # Match lengths
                    min_len = min(len(gt), len(pred))
                    gt = gt[:min_len]
                    pred = pred[:min_len]

                    mae = np.mean(np.abs(gt - pred))
                    report.append(f"  {app}: MAE = {mae:.2f}W")

        report.append("="*60)

        return "\n".join(report)

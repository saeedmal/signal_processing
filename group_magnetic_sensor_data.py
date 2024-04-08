import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import h5py

class GroupMagneticSensorData:
    def __init__(self, group_frequency_amplitude_data: h5py._hl.group.Group):
        """
        Initializes the GroupMagneticSensorData object.

        Args:
        - group_frequency_amplitude_data (h5py.Group): Group containing frequency and amplitude data.
        """
        self.group_ADC = group_frequency_amplitude_data

        self.current_ADC = self._get_current_ADC()
        self.magnetic_ADC = self._get_mag_ADC()
        self.recorded_current_timestamp = self._get_recorded_current_timestamp()
        self.recorded_magnetic_timestamp = self._get_recorded_magnetic_timestamp()
        self.trigger_timestamp = self._get_trigger_timestamp()

        self.amplitude = self.group_ADC.attrs.get('amplitude')
        self.axis = self.group_ADC.attrs.get('axis')
        self.frequency = self.group_ADC.attrs.get('frequency')
        self.sensor_orientation = self.group_ADC.attrs.get('sensor_orientation')
        self.adc_timestamp_offsets = self._get_adc_timestamp_offsets()
        self.timestamp_counts_per_second = self.group_ADC.attrs.get('timestamp_counts_per_s')
        self.current_mA_per_lsb = self.group_ADC.attrs.get('current_mA_per_lsb')
        self.magnetic_uT_per_lsb = self.group_ADC.attrs.get('mag_uT_per_lsb')

    def _get_current_ADC(self) -> np.ndarray:
        """
        Get the current ADC data.

        Returns:
        - np.ndarray: Current ADC data.
        """
        current_ADC_tensor = np.asarray(self.group_ADC["Current"])
        return np.squeeze(current_ADC_tensor, axis=0)

    def _get_mag_ADC(self) -> np.ndarray:
        """
        Get the magnetic ADC data.

        Returns:
        - np.ndarray: Magnetic ADC data.
        """
        mag_ADC_tensor = np.asarray(self.group_ADC["Mag"])
        return np.squeeze(mag_ADC_tensor, axis=0)

    def _get_recorded_current_timestamp(self) -> np.ndarray:
        """
        Get the recorded current timestamps.

        Returns:
        - np.ndarray: Recorded current timestamps.
        """
        return np.asarray(self.group_ADC['Current_Timestamp'])

    def _get_recorded_magnetic_timestamp(self) -> np.ndarray:
        """
        Get the recorded magnetic timestamps.

        Returns:
        - np.ndarray: Recorded magnetic timestamps.
        """
        return np.asarray(self.group_ADC['Mag_Timestamp'])

    def _get_trigger_timestamp(self) -> np.ndarray:
        """
        Get the trigger timestamps.

        Returns:
        - np.ndarray: Trigger timestamps.
        """
        return np.asarray(self.group_ADC['Trigger_Timestamp'])

    def _get_adc_timestamp_offsets(self) -> np.ndarray:
        """
        Get the ADC timestamp offsets.

        Returns:
        - np.ndarray: ADC timestamp offsets.
        """
        return np.asarray(self.group_ADC.attrs.get('adc_timestamp_offsets'))

    def apply_timestamp_offset(self, recorded_timestamp: np.ndarray, channel: int) -> np.ndarray:
        """
        Apply timestamp offset to recorded timestamp.

        Args:
        - recorded_timestamp (np.ndarray): Recorded timestamps.
        - channel (int): Channel number.

        Returns:
        - np.ndarray: Timestamps with offset applied.
        """
        if channel > 2 or channel < 0:
            raise RuntimeError("Invalid channel number outside of 0, 1, 2!")
        offset_time = self.adc_timestamp_offsets[0, channel]
        return recorded_timestamp + offset_time

    def convert_ADC_current_to_measurement(self) -> np.ndarray:
        """
        Convert current ADC to current measurement.

        Returns:
        - np.ndarray: Current measurement.
        """
        return self.current_ADC @ np.diag(self.current_mA_per_lsb)

    def convert_ADC_magnetic_to_measurement(self) -> np.ndarray:
        """
        Convert magnetic ADC to magnetic measurement.

        Returns:
        - np.ndarray: Magnetic measurement.
        """
        return self.magnetic_ADC @ np.diag(self.magnetic_uT_per_lsb)

    @staticmethod
    def find_closest_index(timestamp_record: np.ndarray, value: float) -> int:
        """
        Find the index of the closest value in the timestamp record.

        Args:
        - timestamp_record (np.ndarray): Timestamp record.
        - value (float): Value to find.

        Returns:
        - int: Index of the closest value.
        """
        closest_index = np.abs(timestamp_record - value).argmin()
        return closest_index

    def transfer_sensor_orientation(self, magnetic_measurement: np.ndarray,
                                    corresponding_current: np.ndarray) -> np.ndarray:
        """
        Transfer sensor orientation.

        Args:
        - magnetic_measurement (np.ndarray): Magnetic measurement.
        - corresponding_current (np.ndarray): Corresponding current.

        Returns:
        - np.ndarray: Transferred sensor orientation.
        """
        if not len(self.sensor_orientation) == 6:
            raise RuntimeError("Wrong mapping string is given. Expecting a 6 character string!")
        sign_dict = {"N": -1, "P": 1}
        axis_dict = {"X": 0, "Y": 1, "Z": 2}
        result_matrix = np.zeros((1, 6))
        for index in range(0, 5, 2):
            sensor_index = axis_dict[self.sensor_orientation[index]]
            sign_current = sign_dict[self.sensor_orientation[index + 1]]
            column = int(index / 2)
            corresponding_current_component = sign_current * corresponding_current[column]
            result_matrix[0, column] = corresponding_current_component
            result_matrix[0, column + 3] = magnetic_measurement[sensor_index]
        return result_matrix

    def match_current_mag(self) -> np.ndarray:
        """
        Match current and magnetic measurements.

        Returns:
        - np.ndarray: Matched current and magnetic measurements.
        """
        if len(self.recorded_current_timestamp) > len(self.recorded_magnetic_timestamp):
            raise RuntimeError(
                "Not enough magnetic measurements matching the sampling rate of the current in the coils!")

        magnetic_field_measurements = self.convert_ADC_magnetic_to_measurement()
        current_measurements = self.convert_ADC_current_to_measurement()

        map_current_magnetic = np.zeros((1, 6))
        mag_vector = np.zeros(np.shape(magnetic_field_measurements))
        current_vector = np.zeros(np.shape(current_measurements))

        for channel in range(3):
            actual_times_current = self.apply_timestamp_offset(self.recorded_current_timestamp, channel)
            actual_times_magnetic = self.apply_timestamp_offset(self.recorded_magnetic_timestamp, channel)

            for index in range(len(actual_times_current)):
                time_stamp = actual_times_current[index]
                corresponding_magnetic_index = self.find_closest_index(actual_times_magnetic, time_stamp)
                mag_vector[index, channel] = magnetic_field_measurements[corresponding_magnetic_index, channel]
                current_vector[index, channel] = current_measurements[index, channel]

        for synch_index in range(len(self.recorded_current_timestamp)):
            corresponding_current_magnetic_record = self.transfer_sensor_orientation(
                magnetic_field_measurements[synch_index], current_measurements[synch_index])
            map_current_magnetic = np.vstack((map_current_magnetic, corresponding_current_magnetic_record))
        return map_current_magnetic[1:, :]

    def check_proportional(self, map_currents_magnetics: np.ndarray) -> np.ndarray:
        """
        Calculate magnetic current coefficients.

        Args:
        - map_currents_magnetics (np.ndarray): Array containing mapped current and magnetic measurements.

        Returns:
        - np.ndarray: Array containing magnetic current coefficients.
        """
        num_rows, num_columns = np.shape(map_currents_magnetics)
        mag_current_coefficients = np.zeros((1, 3))

        for index in range(num_rows):
            record_current_mag = map_currents_magnetics[index, :]
            coefficient_x = record_current_mag[3] / record_current_mag[0]
            coefficient_y = record_current_mag[4] / record_current_mag[1]
            coefficient_z = record_current_mag[5] / record_current_mag[2]

            mag_current_coefficients = np.vstack(
                (mag_current_coefficients, np.array([coefficient_x, coefficient_y, coefficient_z])))

        return mag_current_coefficients[1:, :]


    def find_peak_indices_signal(self, signal_type):
        """
        Finds peak indices in the signal.

        Args:
        signal_type (str): Type of signal - 'Current' or 'Mag'.

        Returns:
        numpy.ndarray: Array containing peak indices.
        """
        if signal_type == "Current":
            # Get current signal measurements
            signal_measurements = self.convert_ADC_current_to_measurement()
            signal_measurement_channel = signal_measurements[:, self.axis]
            actual_trigger_time = self.trigger_timestamp
            actual_times_signal = self.apply_timestamp_offset(self.recorded_current_timestamp, self.axis)
        elif signal_type == "Mag":
            # Get magnetic signal measurements
            signal_measurements = self.convert_ADC_magnetic_to_measurement()
            signal_measurement_channel = signal_measurements[:, self.axis]
            actual_trigger_time = self.trigger_timestamp
            actual_times_signal = self.apply_timestamp_offset(self.recorded_magnetic_timestamp, self.axis)
        else:
            raise ValueError("Invalid signal type. Please use either 'Mag' or 'Current'!")

        save_min_max_signal = np.zeros((1, 2))
        number_triggers = actual_trigger_time.shape[0]

        for trigger_index in range(number_triggers - 1):
            phase_zero_time = actual_trigger_time[trigger_index]
            next_phase_zero_time = actual_trigger_time[trigger_index + 1]
            index_phase_zero_current = self.find_closest_index(actual_times_signal, phase_zero_time)
            next_index_phase_zero_current = self.find_closest_index(actual_times_signal, next_phase_zero_time)

            if index_phase_zero_current < next_index_phase_zero_current:
                # Find peak indices within the interval
                max_signal_index = index_phase_zero_current + np.argmax(signal_measurement_channel[index_phase_zero_current:next_index_phase_zero_current])
                min_signal_index = index_phase_zero_current + np.argmin(signal_measurement_channel[index_phase_zero_current:next_index_phase_zero_current])
                indices_tuple = (min_signal_index, max_signal_index)
                min_index = min(indices_tuple)
                max_index = max(indices_tuple)
                save_min_max_signal = np.vstack((save_min_max_signal, np.array([min_index, max_index])))

        return save_min_max_signal[1:, :]
    
    def find_gain_peaks(self, peak_current_indices, peak_magnetic_indices):
        """
        Finds gain peaks from peak indices of current and magnetic signals.

        Args:
        peak_current_indices (numpy.ndarray): Peak indices of current signal.
        peak_magnetic_indices (numpy.ndarray): Peak indices of magnetic signal.

        Returns:
        tuple: Tuple containing mag_current_coefficients and saved_peak_records.
        """
        num_rows_current = peak_current_indices.shape[0]
        num_rows_magnetic = peak_magnetic_indices.shape[0]
        num_cols_magnetic = peak_magnetic_indices.shape[1]
        num_cols_current = peak_current_indices.shape[1]

        if not (num_cols_magnetic == 2 and num_cols_current == 2):
            raise RuntimeError("Peak indices not selected correctly!")

        magnetic_measurements = self.convert_ADC_magnetic_to_measurement()
        current_measurements = self.convert_ADC_current_to_measurement()
        saved_peak_records = np.zeros((1, 6))
        
        for index in range(min(num_rows_current, num_rows_magnetic)):
            first_peak_index_current = int(peak_current_indices[index, 0])
            second_peak_index_current = int(peak_current_indices[index, 1])
            
            first_peak_index_magnetic = int(peak_magnetic_indices[index, 0])
            second_peak_index_magnetic = int(peak_magnetic_indices[index, 1])

            corresponding_current_magnetic_peak_1 = self.transfer_sensor_orientation(
                magnetic_measurements[first_peak_index_magnetic, :],
                current_measurements[first_peak_index_current, :]
            )
            saved_peak_records = np.vstack((saved_peak_records, corresponding_current_magnetic_peak_1))

            corresponding_current_magnetic_peak_2 = self.transfer_sensor_orientation(
                magnetic_measurements[second_peak_index_magnetic, :],
                current_measurements[second_peak_index_current, :]
            )
            saved_peak_records = np.vstack((saved_peak_records, corresponding_current_magnetic_peak_2))

        mag_current_coefficients = self.check_proportional(saved_peak_records[1:, :])
        return mag_current_coefficients, saved_peak_records[1:, :]

    def estimate_gain_and_skewness_from_peaks(self, saved_peaks_magnetic_current):
        """
        Estimates gain and skewness from saved peaks of magnetic and current signals.

        Args:
        saved_peaks_magnetic_current (numpy.ndarray): Array containing saved peaks of magnetic and current signals.

        Returns:
        tuple: Tuple containing estimated_gain, estimated_skewness, mse, mae, and inlier_ratio.
        """
        num_samples = saved_peaks_magnetic_current.shape[0]
        train_sample_count = int(0.8 * num_samples)
        
        input_currents_train = saved_peaks_magnetic_current[:train_sample_count, self.axis].reshape(-1, 1)
        output_magnetic_train = saved_peaks_magnetic_current[:train_sample_count, 3 + self.axis].reshape(-1, 1)

        input_currents_test = saved_peaks_magnetic_current[train_sample_count:, self.axis].reshape(-1, 1)
        output_magnetic_test = saved_peaks_magnetic_current[train_sample_count:, 3 + self.axis].reshape(-1, 1)

        ransac_estimator = RANSACRegressor()
        ransac_estimator.fit(input_currents_train, output_magnetic_train)
        
        estimated_gain = ransac_estimator.estimator_.coef_
        estimated_skewness = ransac_estimator.estimator_.intercept_

        output_magnetic_predicted = ransac_estimator.predict(input_currents_test)

        mse = mean_squared_error(output_magnetic_test, output_magnetic_predicted)
        mae = mean_absolute_error(output_magnetic_test, output_magnetic_predicted)
        inlier_ratio = ransac_estimator.inlier_mask_.sum() / (0.8 * num_samples)

        return estimated_gain, estimated_skewness, mse, mae, inlier_ratio

    def compare_sample_times(self, record_peak_current, record_peak_magnetic):
        """
        Compares sample times between peak records of current and magnetic signals.

        Args:
        record_peak_current (numpy.ndarray): Peak records of current signal.
        record_peak_magnetic (numpy.ndarray): Peak records of magnetic signal.

        Returns:
        numpy.ndarray: Array containing the difference in sample times.
        """
        num_rows_current = record_peak_current.shape[0]
        num_rows_magnetic = record_peak_magnetic.shape[0]
        min_rows = min(num_rows_current, num_rows_magnetic)

        # Compare sample times between current and magnetic peak records
        return record_peak_magnetic[:min_rows, :] - record_peak_current[:min_rows, :]
    
    def frequency_response(self) -> list:
        """
        Calculate the frequency response of the magnetic sensor.
        
        Returns:
            list: A list containing dictionaries with frequency response data for each coil.
        """
        magnetic_current_matched = self.match_current_mag()
        save_frequency_response = []
        for channel in range(3):
            actual_times_current = self.apply_timestamp_offset(self.recorded_current_timestamp, channel)
            dft_current_signal = np.fft.fft(magnetic_current_matched[:, channel])
            sample_number = np.shape(dft_current_signal)[0]
            dft_magnetic_signal = np.fft.fft(magnetic_current_matched[:, channel + 3])
            frequency_response = np.abs(dft_magnetic_signal[:sample_number]/dft_current_signal)
            frequency_axis = np.fft.fftfreq(sample_number, d=actual_times_current[1] - actual_times_current[0])
            save_frequency_response.append({"coil": channel, "frequency_axis": frequency_axis,
                                             "frequency_response": frequency_response})
        return save_frequency_response


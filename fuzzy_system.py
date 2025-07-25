import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def build_fuzzy_system():
    """
    Membangun sistem fuzzy untuk menentukan tingkat kerusakan daun tebu
    berdasarkan damage_area dan color_intensity.
    """

    # Definisi variabel fuzzy input
    damage_area = ctrl.Antecedent(np.arange(0, 101, 1), 'damage_area')
    color_intensity = ctrl.Antecedent(np.arange(0, 101, 1), 'color_intensity')

    # Definisi variabel fuzzy output
    damage_level = ctrl.Consequent(np.arange(0, 101, 1), 'damage_level')

    # Fungsi keanggotaan input damage_area
    damage_area['low'] = fuzz.trimf(damage_area.universe, [0, 0, 50])
    damage_area['medium'] = fuzz.trimf(damage_area.universe, [20, 50, 80])
    damage_area['high'] = fuzz.trimf(damage_area.universe, [50, 100, 100])

    # Fungsi keanggotaan input color_intensity
    color_intensity['low'] = fuzz.trimf(color_intensity.universe, [0, 0, 50])
    color_intensity['medium'] = fuzz.trimf(color_intensity.universe, [20, 50, 80])
    color_intensity['high'] = fuzz.trimf(color_intensity.universe, [50, 100, 100])

    # Fungsi keanggotaan output damage_level
    damage_level['sehat'] = fuzz.trimf(damage_level.universe, [0, 0, 10])
    damage_level['mosaik'] = fuzz.trimf(damage_level.universe, [5, 15, 25])
    damage_level['busuk_merah'] = fuzz.trimf(damage_level.universe, [20, 30, 40])
    damage_level['karat'] = fuzz.trimf(damage_level.universe, [30, 50, 60])
    damage_level['kuning'] = fuzz.trimf(damage_level.universe, [60, 80, 100])

    # Aturan fuzzy
    rules = [
        ctrl.Rule(damage_area['low'] & color_intensity['low'], damage_level['sehat']),
        ctrl.Rule(damage_area['medium'] & color_intensity['low'], damage_level['mosaik']),
        ctrl.Rule(damage_area['medium'] & color_intensity['medium'], damage_level['busuk_merah']),
        ctrl.Rule(damage_area['high'] & color_intensity['medium'], damage_level['karat']),
        ctrl.Rule(damage_area['high'] & color_intensity['high'], damage_level['kuning']),
    ]

    # Sistem kontrol fuzzy
    damage_ctrl = ctrl.ControlSystem(rules)
    return damage_ctrl


def evaluate_fuzzy_system(damage_area_value, color_intensity_value):
    """
    Menghitung tingkat kerusakan (damage level) berdasarkan input damage_area dan color_intensity
    menggunakan sistem fuzzy yang telah dibangun.
    
    Parameters:
    - damage_area_value: int/float (0-100)
    - color_intensity_value: int/float (0-100)
    
    Returns:
    - damage_level_value: float, output defuzzifikasi dari sistem fuzzy (0-100)
    """

    damage_ctrl = build_fuzzy_system()
    simulation = ctrl.ControlSystemSimulation(damage_ctrl)

    simulation.input['damage_area'] = damage_area_value
    simulation.input['color_intensity'] = color_intensity_value

    simulation.compute()

    return simulation.output['damage_level']


# Pemetaan label CNN ke tingkat kerusakan dalam bentuk string dan numerik
DAMAGE_MAPPING = {
    'sehat': '0%',
    'mosaik': '10%',
    'busuk merah': '25%',
    'karat': '45%',
    'kuning': '75%'
}

DAMAGE_NUMERIC = {
    'sehat': 0,
    'mosaik': 10,
    'busuk merah': 25,
    'karat': 45,
    'kuning': 75
}


def get_damage_level(predicted_label):
    """
    Mengembalikan tingkat kerusakan dalam bentuk string persentase berdasarkan label CNN.
    
    Jika label tidak dikenali, mengembalikan 'Tidak Diketahui'.
    """
    return DAMAGE_MAPPING.get(predicted_label.lower(), 'Tidak Diketahui')


def get_damage_numeric(predicted_label):
    """
    Mengembalikan tingkat kerusakan dalam bentuk angka (integer) berdasarkan label CNN.
    
    Jika label tidak dikenali, mengembalikan 0.
    """
    return DAMAGE_NUMERIC.get(predicted_label.lower(), 0)

"""
global_result.py  (ubicado en result/global/)

Script orquestador para generar TODOS los resultados del proyecto:

1. result/classification/
   - summary_classification_runs.csv
   - gráficos comparativos entre modelos por dispositivo
   - gráficos comparativos RTX 4080 vs A100 por modelo

2. result/detection/
   - summary_detection_runs.csv
   - gráficos comparativos entre modelos por dispositivo
   - gráficos comparativos RTX 4080 vs A100 por modelo

3. result/segmentation/
   - summary_segmentation_runs.csv
   - gráficos comparativos entre modelos por dispositivo
   - gráficos comparativos RTX 4080 vs A100 por modelo

4. result/global/
   - summary_all_runs.csv
   - gráficos de comparación global RTX 4080 vs A100 por modelo/tarea

Ejecución recomendada desde la raíz del proyecto:

    export PYTHONPATH=./src
    python result/global/global_result.py
"""

from __future__ import annotations

from pathlib import Path

# -------------------------------------------------------------------
# Imports con try/except por si alguna parte aún no existe
# -------------------------------------------------------------------

try:
    from classification.classification_results import main as classification_main
except ImportError:
    classification_main = None

try:
    from detection.detection_results import main as detection_main
except ImportError:
    detection_main = None

try:
    from segmentation.segmentation_results import main as segmentation_main
except ImportError:
    segmentation_main = None

try:
    # resumen global (utils/utils_global.py)
    from utils.utils_global import main as global_main
except ImportError:
    global_main = None


def main() -> None:
    # Este archivo está en result/global/, así que el proyecto está 2 niveles arriba
    project_root = Path(__file__).resolve().parents[2]
    print(f"[INFO] Proyecto raíz detectado en: {project_root}")

    # 1) Resultados de CLASIFICACIÓN
    if classification_main is not None:
        print("\n[STEP] Generando resultados de CLASIFICACIÓN...")
        try:
            classification_main()
        except Exception as e:
            print(f"[ERROR] Falló classification_main(): {e}")
    else:
        print("[WARN] No se pudo importar classification.classification_results")

    # 2) Resultados de DETECCIÓN
    if detection_main is not None:
        print("\n[STEP] Generando resultados de DETECCIÓN...")
        try:
            detection_main()
        except Exception as e:
            print(f"[ERROR] Falló detection_main(): {e}")
    else:
        print("[WARN] No se pudo importar detection.detection_results")

    # 3) Resultados de SEGMENTACIÓN
    if segmentation_main is not None:
        print("\n[STEP] Generando resultados de SEGMENTACIÓN...")
        try:
            segmentation_main()
        except Exception as e:
            print(f"[ERROR] Falló segmentation_main(): {e}")
    else:
        print("[WARN] No se pudo importar segmentation.segmentation_results")

    # 4) Resumen GLOBAL (todas las tareas / GPUs / modelos)
    if global_main is not None:
        print("\n[STEP] Generando resultados GLOBALES...")
        try:
            global_main()
        except Exception as e:
            print(f"[ERROR] Falló utils.utils_global.main(): {e}")
    else:
        print("[WARN] No se pudo importar utils.utils_global")

    print("\n[INFO] Proceso global de resultados COMPLETADO.")


if __name__ == "__main__":
    main()

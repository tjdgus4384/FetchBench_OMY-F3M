
from .fetch.fetch_base import FetchBase
from .fetch.fetch_ptd import FetchPointCloudBase
from .fetch.fetch_naive import FetchNaive

# CuRobo
from .fetch.fetch_mesh_curobo import FetchMeshCurobo
from .fetch.repeat.fetch_mesh_curobo_rep import FetchMeshCuroboRep
from .fetch.fetch_ptd_curobo import FetchPtdCurobo
from .fetch.repeat.fetch_ptd_curobo_rep import FetchPtdCuroboRep

from .fetch.fetch_mesh_curobo_datagen import FetchCuroboDataGen

# Imit Beta
from .fetch.imit.fetch_ptd_imit_e2e import FetchPtdImitE2E
from .fetch.imit.fetch_ptd_imit_two_stage import FetchPtdImitTwoStage
from .fetch.imit.fetch_ptd_imit_curobo_cgn import FetchPtdImitCuroboCGN

isaacgym_task_map = {
    "FetchBase": FetchBase,
    "FetchPointCloudBase": FetchPointCloudBase,

    "FetchNaive": FetchNaive,
    "FetchMeshCurobo": FetchMeshCurobo,
    "FetchMeshCuroboRep": FetchMeshCuroboRep,

    "FetchPtdCurobo": FetchPtdCurobo,
    "FetchPtdCuroboRep": FetchPtdCuroboRep,

    "FetchCuroboDataGen": FetchCuroboDataGen,

    "FetchPtdImitE2E": FetchPtdImitE2E,
    "FetchPtdImitTwoStage": FetchPtdImitTwoStage,
    "FetchPtdImitCuroboCGN": FetchPtdImitCuroboCGN
}


try:
    # OMPL
    try:
        from .fetch.fetch_mesh_pyompl import FetchMeshPyompl
    except: FetchMeshPyompl = None
    try:
        from .fetch.repeat.fetch_mesh_pyompl_rep import FetchMeshPyomplRep
    except: FetchMeshPyomplRep = None
    try:
        from .fetch.fetch_ptd_pyompl import FetchPtdPyompl
    except: FetchPtdPyompl = None
    try:
        from .fetch.repeat.fetch_ptd_pyompl_rep import FetchPtdPyomplRep
    except: FetchPtdPyomplRep = None

    # Cabinet
    try:
        from .fetch.fetch_ptd_cabinet import FetchPtdCabinet
    except: FetchPtdCabinet = None
    try:
        from .fetch.fetch_ptd_cabinet_cgn_beta import FetchPtdCabinetCGNBeta
    except: FetchPtdCabinetCGNBeta = None

    # Contact_Graspnet_Pytorch
    from .fetch.fetch_ptd_curobo_cgn_beta import FetchPtdCuroboCGNBeta
    try:
        from .fetch.fetch_ptd_pyompl_cgn_beta import FetchPtdPyomplCGNBeta
    except: FetchPtdPyomplCGNBeta = None
    from .fetch.fetch_mesh_curobo_cgn_beta import FetchMeshCuroboPtdCGNBeta
    try:
        from .fetch.fetch_mesh_curobo_graspgen import FetchMeshCuroboGraspGen
    except: FetchMeshCuroboGraspGen = None
    try:
        from .fetch.fetch_mesh_pyompl_cgn_beta import FetchMeshPyomplPtdCGNBeta
    except: FetchMeshPyomplPtdCGNBeta = None

    from .fetch.repeat.fetch_mesh_curobo_cgn_beta_rep import FetchMeshCuroboPtdCGNBetaRep
    from .fetch.repeat.fetch_ptd_curobo_cgn_beta_rep import FetchPtdCuroboCGNBetaRep
    try:
        from .fetch.repeat.fetch_ptd_pyompl_cgn_beta_rep import FetchPtdPyomplCGNBetaRep
    except: FetchPtdPyomplCGNBetaRep = None

    isaacgym_task_map = {
        "FetchBase": FetchBase,
        "FetchPointCloudBase": FetchPointCloudBase,

        "FetchNaive": FetchNaive,
        "FetchMeshCurobo": FetchMeshCurobo,
        "FetchMeshPyompl": FetchMeshPyompl,
        "FetchMeshCuroboRep": FetchMeshCuroboRep,
        "FetchMeshPyomplRep": FetchMeshPyomplRep,

        "FetchPtdCurobo": FetchPtdCurobo,
        "FetchPtdPyompl": FetchPtdPyompl,

        "FetchCuroboDataGen": FetchCuroboDataGen,

        "FetchPtdCuroboRep": FetchPtdCuroboRep,
        "FetchPtdPyomplRep": FetchPtdPyomplRep,

        "FetchPtdCuroboCGNBeta": FetchPtdCuroboCGNBeta,
        "FetchPtdPyomplCGNBeta": FetchPtdPyomplCGNBeta,
        "FetchMeshCuroboPtdCGNBeta": FetchMeshCuroboPtdCGNBeta,
        "FetchMeshCuroboGraspGen": FetchMeshCuroboGraspGen,
        "FetchMeshPyomplPtdCGNBeta": FetchMeshPyomplPtdCGNBeta,

        "FetchPtdCabinet": FetchPtdCabinet,
        "FetchPtdCabinetCGNBeta": FetchPtdCabinetCGNBeta,

        "FetchMeshCuroboPtdCGNBetaRep": FetchMeshCuroboPtdCGNBetaRep,
        "FetchPtdCuroboCGNBetaRep": FetchPtdCuroboCGNBetaRep,
        "FetchPtdPyomplCGNBetaRep": FetchPtdPyomplCGNBetaRep,

        "FetchPtdImitE2E": FetchPtdImitE2E,
        "FetchPtdImitTwoStage": FetchPtdImitTwoStage,
        "FetchPtdImitCuroboCGN": FetchPtdImitCuroboCGN
    }

except:
    print("============================================================")
    print("Import Error: Additional Method Excluded.")


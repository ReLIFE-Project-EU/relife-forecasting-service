
def _norm_surface_name(name: str) -> str:
    """Normalizza i nomi delle superfici per il lookup."""
    return " ".join(name.lower().split())


U_VALUES_BY_CLASS = {
    "A": {
        _norm_surface_name("Roof surface"): 0.5,
        _norm_surface_name("Opaque north surface"): 0.6,
        _norm_surface_name("Opaque south surface"): 0.6,
        _norm_surface_name("Opaque east surface"): 0.6,
        _norm_surface_name("Opaque west surface"): 0.6,
        _norm_surface_name("Slab to ground"): 1.5,
        _norm_surface_name("Transparent north surface"): 3.2,
        _norm_surface_name("Transparent south surface"): 3.2,
        _norm_surface_name("Transparent east surface"): 3.2,
        _norm_surface_name("Transparent west surface"): 3.2,
    },
    "B": {
        _norm_surface_name("Roof surface"): 0.4,
        _norm_surface_name("Opaque north surface"): 0.5,
        _norm_surface_name("Opaque south surface"): 0.5,
        _norm_surface_name("Opaque east surface"): 0.5,
        _norm_surface_name("Opaque west surface"): 0.5,
        _norm_surface_name("Slab to ground"): 1.0,
        _norm_surface_name("Transparent north surface"): 3.0,
        _norm_surface_name("Transparent south surface"): 3.0,
        _norm_surface_name("Transparent east surface"): 3.0,
        _norm_surface_name("Transparent west surface"): 3.0,
    },
    "C": {
        _norm_surface_name("Roof surface"): 0.38,
        _norm_surface_name("Opaque north surface"): 0.44,
        _norm_surface_name("Opaque south surface"): 0.44,
        _norm_surface_name("Opaque east surface"): 0.44,
        _norm_surface_name("Opaque west surface"): 0.44,
        _norm_surface_name("Slab to ground"): 0.38,
        _norm_surface_name("Transparent north surface"): 2.8,
        _norm_surface_name("Transparent south surface"): 2.8,
        _norm_surface_name("Transparent east surface"): 2.8,
        _norm_surface_name("Transparent west surface"): 2.8,
    },
    "D": {
        _norm_surface_name("Roof surface"): 0.35,
        _norm_surface_name("Opaque north surface"): 0.33,
        _norm_surface_name("Opaque south surface"): 0.33,
        _norm_surface_name("Opaque east surface"): 0.33,
        _norm_surface_name("Opaque west surface"): 0.33,
        _norm_surface_name("Slab to ground"): 0.35,
        _norm_surface_name("Transparent north surface"): 2.6,
        _norm_surface_name("Transparent south surface"): 2.6,
        _norm_surface_name("Transparent east surface"): 2.6,
        _norm_surface_name("Transparent west surface"): 2.6,
    },
}
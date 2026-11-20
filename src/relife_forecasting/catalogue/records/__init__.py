from ..spec import ArchetypeSpec
from . import (
    at,
    be,
    bg,
    cy,
    cz,
    de,
    dk,
    ee,
    es,
    fi,
    fr,
    gr,
    hr,
    hu,
    ie,
    it,
    lt,
    lu,
    lv,
    mt,
    nl,
    pl,
    pt,
    ro,
    se,
    si,
    sk,
)

COUNTRIES = (
    at.RECORDS,
    be.RECORDS,
    bg.RECORDS,
    hr.RECORDS,
    cy.RECORDS,
    cz.RECORDS,
    dk.RECORDS,
    ee.RECORDS,
    fi.RECORDS,
    fr.RECORDS,
    de.RECORDS,
    gr.RECORDS,
    hu.RECORDS,
    ie.RECORDS,
    it.RECORDS,
    lv.RECORDS,
    lt.RECORDS,
    lu.RECORDS,
    mt.RECORDS,
    nl.RECORDS,
    pl.RECORDS,
    pt.RECORDS,
    ro.RECORDS,
    sk.RECORDS,
    si.RECORDS,
    es.RECORDS,
    se.RECORDS,
)


def source_records() -> dict[str, ArchetypeSpec]:
    records = {}
    for country in COUNTRIES:
        for name, spec in country.items():
            if name in records:
                raise ValueError(f"Duplicate archetype {name}")
            records[name] = spec.copy()
    return records

# Extra branch lists needed by the MAPLE selection that are not in makedf/branches.py.

crtpmtbranches = [
    "rec.crtpmt_matches.flashGateTime",
    "rec.crtpmt_matches.flashPE",
    "rec.crtpmt_matches.flashPosition.x",
    "rec.crtpmt_matches.flashPosition.y",
    "rec.crtpmt_matches.flashPosition.z",
    "rec.crtpmt_matches.flashClassification",
]

slcchargecenterbranches = [
    "rec.slc.charge_center.x",
    "rec.slc.charge_center.y",
    "rec.slc.charge_center.z",
]

shwenergybranches = [
    "rec.slc.reco.pfp.shw.plane.2.energy",
]

# Truth branches for the MAPLE truth classification (classification_type_MC)
mapleprimbranches = [
    "rec.mc.nu.prim.pdg",
    "rec.mc.nu.prim.G4ID",
    "rec.mc.nu.prim.cryostat",
    "rec.mc.nu.prim.length",
    "rec.mc.nu.prim.end.x", "rec.mc.nu.prim.end.y", "rec.mc.nu.prim.end.z",
    "rec.mc.nu.prim.genp.x", "rec.mc.nu.prim.genp.y", "rec.mc.nu.prim.genp.z",
    "rec.mc.nu.prim.plane.0.2.visE",
    "rec.mc.nu.prim.plane.1.2.visE",
]

mapletruepartbranches = [
    "rec.true_particles.pdg",
    "rec.true_particles.parent",
    "rec.true_particles.cryostat",
    "rec.true_particles.end.x", "rec.true_particles.end.y", "rec.true_particles.end.z",
    "rec.true_particles.plane.0.2.visE",
    "rec.true_particles.plane.1.2.visE",
]

maplemcbranches = [
    "rec.mc.nu.E",
    "rec.mc.nu.pdg",
    "rec.mc.nu.iscc",
    "rec.mc.nu.genie_mode",
    "rec.mc.nu.position.x",
    "rec.mc.nu.position.y",
    "rec.mc.nu.position.z",
    "rec.mc.nu.baseline",
    "rec.mc.nu.time",
]

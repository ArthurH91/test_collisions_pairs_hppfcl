import numpy as np
import hppfcl
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


RED = np.array([249, 136, 126, 125]) / 255
RED_FULL = np.array([249, 136, 126, 255]) / 255

GREEN = np.array([170, 236, 149, 125]) / 255
GREEN_FULL = np.array([170, 236, 149, 255]) / 255

BLUE = np.array([144, 169, 183, 125]) / 255
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

YELLOW = np.array([1, 1, 0, 0.5])
YELLOW_FULL = np.array([1, 1, 0, 1.])

BLACK = np.array([0, 0, 0, 0.5])
BLACK_FULL = np.array([0, 0, 0, 1.])

GREY = np.array([128,128,128,125])/255
GREY_FULL = np.array([128,128,128,255])/255

### Creating a robot model

rmodel = pin.Model()
cmodel = pin.GeometryModel()


r_sphere = 1.5
r1 = 0.5
halfLength = 0.5
x, y, z = 0.5, 0.2, 0.6


rmodel = pin.Model()
cmodel = pin.GeometryModel()
geometries = [
    hppfcl.Sphere(r_sphere),
    hppfcl.Sphere(r1),
    hppfcl.Capsule(r1, halfLength),
    hppfcl.Cylinder(r1, halfLength),
    hppfcl.Ellipsoid(x,y,z),
    hppfcl.Box(x,y,z)
]

placements = [
    pin.SE3(np.eye(3), np.array([0, 0, 0])),
    pin.SE3(pin.SE3.Random().rotation, np.array([-0.2, -1.5, -0.0])),
    pin.SE3(pin.SE3.Random().rotation, np.array([0.3, 1.5, 0])),
    pin.SE3(pin.SE3.Random().rotation, np.array([-1.3, 0.3, 0])),
    pin.SE3(pin.SE3.Random().rotation, np.array([1.5, 0.3, 0.3])),
    pin.SE3(pin.SE3.Random().rotation, np.array([1.0, -1.2, 0.3])),
]

colors = [RED, BLUE, GREEN, YELLOW, GREY, BLACK]

req = hppfcl.DistanceRequest()
res = hppfcl.DistanceResult()

for i, geom in enumerate(geometries):
    placement = placements[i]
    geom_obj = pin.GeometryObject("obj" + str(i), 0, 0, placement, geom)
    geom_obj.meshColor = colors[i]
    cmodel.addGeometryObject(geom_obj)
    
    
for geom_obj in cmodel.geometryObjects:
    pos_sphere = cmodel.geometryObjects[cmodel.getGeometryId("obj0")].placement
    geom_sphere = cmodel.geometryObjects[cmodel.getGeometryId("obj0")].geometry
    if not "obj0" in geom_obj.name:
        print(geom_obj.name)
        dist = hppfcl.distance(
            geom_sphere, 
            pos_sphere,
            geom_obj.geometry,
            geom_obj.placement,
            req,
            res
        )
        cp1 = res.getNearestPoint1()
        placement_cp1 = pin.SE3(np.eye(3), cp1)

        cp2 = res.getNearestPoint2()
        placement_cp2 = pin.SE3(np.eye(3), cp2)
        print(f"cp1: {cp1}")
        print(f"cp2: {cp2}")
        print(f"dist : {dist}")
        print(f"np.linalg.norm(cp2-cp1): {np.linalg.norm(cp2-cp1)}")
        print("-----------------")
        geom_cp = hppfcl.Sphere(0.05)
        cp1_geom = pin.GeometryObject(geom_obj.name + "_cp1", 0, 0, placement_cp1, geom_cp)
        cp2_geom = pin.GeometryObject(geom_obj.name + "_cp2", 0, 0, placement_cp2, geom_cp)
        cp1_geom.meshColor = BLACK_FULL
        cp2_geom.meshColor = BLACK_FULL
        cmodel.addGeometryObject(cp1_geom)
        cmodel.addGeometryObject(cp2_geom)

        
rdata = rmodel.createData()
cdata = cmodel.createData()

pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

viz = MeshcatVisualizer(rmodel, cmodel,cmodel)
viz.initViewer(open=False)
viz.loadViewerModel()
q0 = pin.neutral(rmodel)
viz.display(q0)

print(cmodel)
input()
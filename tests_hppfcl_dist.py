import unittest

import numpy as np

import pinocchio as pin
import hppfcl


class TestCollisions(unittest.TestCase):
    """This class is made to test the collisions between primitives pairs such as sphere-sphere. The collisions shapes are from hppfcl."""

    def test_sphere_nearest_points_collision_with_other_shapes(self):
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
            hppfcl.Ellipsoid(x, y, z),
            hppfcl.Box(x, y, z),
        ]

        placements = [
            pin.SE3(np.eye(3), np.array([0, 0, 0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([-0.2, -1.5, -0.0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([0.3, 1.5, 0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([-1.3, 0.3, 0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([1.5, 0.3, 0.3])),
            pin.SE3(pin.SE3.Random().rotation, np.array([1.0, -1.2, 0.3])),
        ]
        

        for i, geom in enumerate(geometries):
            placement = placements[i]
            geom_obj = pin.GeometryObject("obj" + str(i), 0, 0, placement, geom)
            cmodel.addGeometryObject(geom_obj)
        
        
        for geom_obj in cmodel.geometryObjects:
            pos_sphere = cmodel.geometryObjects[cmodel.getGeometryId("obj0")].placement
            geom_sphere = cmodel.geometryObjects[cmodel.getGeometryId("obj0")].geometry
            if not "obj0" in geom_obj.name:
                req = hppfcl.DistanceRequest()
                res = hppfcl.DistanceResult()

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

                self.assertFalse(np.array_equal(cp1, cp2), f"The nearest points computed for {geom_obj.geometry} are equal. It is likely that they are not computed properly.")
                self.assertAlmostEqual(np.linalg.norm(dist),np.linalg.norm(cp2-cp1),msg= f"The distance computed for {geom_obj.geometry} has not been computed properly.") 

    def test_capsule_nearest_points_collision_with_other_shapes(self):
        ### Creating a robot model

        rmodel = pin.Model()
        cmodel = pin.GeometryModel()


        r_caps = 1.5
        halfLength_Caps = 3
        r1 = 0.5
        halfLength = 0.5
        x, y, z = 0.5, 0.2, 0.6

        rmodel = pin.Model()
        cmodel = pin.GeometryModel()
        geometries = [
            hppfcl.Capsule(r_caps, halfLength_Caps),
            hppfcl.Sphere(r1),
            hppfcl.Capsule(r1, halfLength),
            hppfcl.Cylinder(r1, halfLength),
            hppfcl.Ellipsoid(x,y,z),
            hppfcl.Box(x,y,z),
            hppfcl.Sphere(r1),
            hppfcl.Capsule(r1, halfLength),
            hppfcl.Cylinder(r1, halfLength),
            hppfcl.Ellipsoid(x,y,z),
            hppfcl.Box(x,y,z),
        ]

        placements = [
            pin.SE3(np.eye(3), np.array([0, 0, 0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([-0.2, -1.5, -0.0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([0.5, 1.5, 0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([-1.3, 0.1, 0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([1.5, 0.9, 0.3])),
            pin.SE3(pin.SE3.Random().rotation, np.array([1.0, -1.2, 0.3])),
            pin.SE3(pin.SE3.Random().rotation, np.array([-0.2, -1.5, 2.2])),
            pin.SE3(pin.SE3.Random().rotation, np.array([-0.2, -1.5, 2.0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([0.5, 1.5, 2.0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([-1.3, 0.1, 2.0])),
            pin.SE3(pin.SE3.Random().rotation, np.array([1.5, 0.9, 2.3])),
            pin.SE3(pin.SE3.Random().rotation, np.array([.5, -1.5, 2.3])),
        ]
        

        for i, geom in enumerate(geometries):
            placement = placements[i]
            geom_obj = pin.GeometryObject("obj" + str(i), 0, 0, placement, geom)
            cmodel.addGeometryObject(geom_obj)
        
        
        for geom_obj in cmodel.geometryObjects:
            pos_ref = cmodel.geometryObjects[cmodel.getGeometryId("obj0")].placement
            geom_ref = cmodel.geometryObjects[cmodel.getGeometryId("obj0")].geometry
            if not "obj0" in geom_obj.name:
                req = hppfcl.DistanceRequest()
                res = hppfcl.DistanceResult()

                dist = hppfcl.distance(
                    geom_ref, 
                    pos_ref,
                    geom_obj.geometry,
                    geom_obj.placement,
                    req,
                    res
                )
                cp1 = res.getNearestPoint1()
                cp2 = res.getNearestPoint2()

                self.assertFalse(np.array_equal(cp1, cp2), f"The nearest points computed for {geom_obj.geometry} are equal. It is likely that they are not computed properly.")
                self.assertAlmostEqual(np.linalg.norm(dist),np.linalg.norm(cp2-cp1),msg= f"The distance computed for {geom_obj.geometry} has not been computed properly.") 


    def test_cylinder_nearest_points_collision_with_other_shapes(self):
            ### Creating a robot model

            rmodel = pin.Model()
            cmodel = pin.GeometryModel()



            r_cyl = 1.5
            halfLength_cyl = 5
            r1 = 0.5
            halfLength = 0.5
            x, y, z = 0.5, 0.2, 0.6

            rmodel = pin.Model()
            cmodel = pin.GeometryModel()
            geometries = [
                hppfcl.Cylinder(r_cyl, halfLength_cyl),
                hppfcl.Sphere(r1),
                hppfcl.Capsule(r1, halfLength),
                hppfcl.Cylinder(r1, halfLength),
                hppfcl.Ellipsoid(x,y,z),
                hppfcl.Box(x,y,z),
                hppfcl.Sphere(r1),
                hppfcl.Capsule(r1, halfLength),
                hppfcl.Cylinder(r1, halfLength),
                hppfcl.Ellipsoid(x,y,z),
                hppfcl.Box(x,y,z),
            ]

            placements = [
                pin.SE3(np.eye(3), np.array([0, 0, 0])),
                pin.SE3(pin.SE3.Random().rotation, np.array([-0.2, -1.5, -0.0])),
                pin.SE3(pin.SE3.Random().rotation, np.array([0.5, 1.5, 0])),
                pin.SE3(pin.SE3.Random().rotation, np.array([-1.3, 0.1, 0])),
                pin.SE3(pin.SE3.Random().rotation, np.array([1.5, 0.9, 0.3])),
                pin.SE3(pin.SE3.Random().rotation, np.array([1.0, -1.2, 0.3])),
                pin.SE3(pin.SE3.Random().rotation, np.array([-0.2, -1.5, 2.2])),
                pin.SE3(pin.SE3.Random().rotation, np.array([0.8, -1.5, 2.0])),
                pin.SE3(pin.SE3.Random().rotation, np.array([0.5, 1.5, 2.0])),
                pin.SE3(pin.SE3.Random().rotation, np.array([-1.3, 0.1, 2.0])),
                pin.SE3(pin.SE3.Random().rotation, np.array([1.5, 0.9, 2.3])),
                pin.SE3(pin.SE3.Random().rotation, np.array([.5, -1.5, 2.3])),
            ]
            

            for i, geom in enumerate(geometries):
                placement = placements[i]
                geom_obj = pin.GeometryObject("obj" + str(i), 0, 0, placement, geom)
                cmodel.addGeometryObject(geom_obj)
            
            
            for geom_obj in cmodel.geometryObjects:
                pos_ref = cmodel.geometryObjects[cmodel.getGeometryId("obj0")].placement
                geom_ref = cmodel.geometryObjects[cmodel.getGeometryId("obj0")].geometry
                if not "obj0" in geom_obj.name:
                    req = hppfcl.DistanceRequest()
                    res = hppfcl.DistanceResult()

                    dist = hppfcl.distance(
                        geom_ref, 
                        pos_ref,
                        geom_obj.geometry,
                        geom_obj.placement,
                        req,
                        res
                    )
                    cp1 = res.getNearestPoint1()
                    cp2 = res.getNearestPoint2()

                    self.assertFalse(np.array_equal(cp1, cp2), f"The nearest points computed for {geom_obj.geometry} are equal. It is likely that they are not computed properly.")
                    self.assertAlmostEqual(np.linalg.norm(dist),np.linalg.norm(cp2-cp1), msg=f"The distance computed for {geom_obj.geometry} has not been computed properly.") 




if __name__ == "__main__":
    unittest.main()

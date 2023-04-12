import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import math
from easy_mesh_vtk import *
import pandas as pd

class Easy_Landmark(object):
    def __init__(self, filename = None, warning=False):
        self.warning = warning
        self.reader = None
        self.vtkPolyData = None
        self.points = np.array([])
        self.point_attributes = dict()
        self.filename = filename
        if self.filename != None:
            if self.filename[-3:].lower() == 'vtp':
                self.read_vtp(self.filename)


    def get_landmark_data_from_vtkPolyData(self):
        data = self.vtkPolyData

        n_points = data.GetNumberOfPoints()
        mesh_points = np.zeros([n_points, 3], dtype='float32')

        for i in range(n_points):
            mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)

        self.points = mesh_points

        #read point arrays
        for i_attribute in range(self.vtkPolyData.GetPointData().GetNumberOfArrays()):
            self.load_point_attributes(self.vtkPolyData.GetPointData().GetArrayName(i_attribute), self.vtkPolyData.GetPointData().GetArray(i_attribute).GetNumberOfComponents())


    def read_fcsv(self, fcsv_filename, landmark_name_list, header=None, skiprows=3, check_name=True):
        '''
        read fcsv, a csv from 3D Slicer for landmarks
        '''
        self.filename = fcsv_filename
        lmk_df = pd.read_csv(self.filename, header=header, skiprows=skiprows)
        num_landmarks = len(lmk_df)
        landmarks = np.zeros([num_landmarks, 3])

        if check_name:
            i = 0
            for i_name in landmark_name_list:
                landmarks[i, :] = lmk_df.loc[lmk_df[11]==i_name][[1, 2, 3]].values
                i += 1
        else:
            landmarks = lmk_df[[1, 2, 3]].values

        self.points = landmarks


    def read_vtp(self, vtp_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.points
            self.point_attributes
        '''
        self.filename = vtp_filename
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_filename)
        reader.Update()
        self.reader = reader

        data = reader.GetOutput()
        self.vtkPolyData = data
        self.get_landmark_data_from_vtkPolyData()


    def load_point_attributes(self, attribute_name, dim):
        self.point_attributes[attribute_name] = np.zeros([self.points.shape[0], dim])
        try:
            if dim == 1:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetValue(i)
            elif dim == 2:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 1)
            elif dim == 3:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 1)
                    self.point_attributes[attribute_name][i, 2] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 2)
        except:
            if self.warning:
                print('No cell attribute named "{0}" in file: {1}'.format(attribute_name, self.filename))


    def update_vtkPolyData(self):
        '''
        call this function when manipulating self.points
        '''
        vtkPolyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()

        points.SetData(numpy_to_vtk(self.points))
        vtkPolyData.SetPoints(points)

        #update point_attributes
        for i_key in self.point_attributes.keys():
            point_attribute = vtk.vtkDoubleArray()
            point_attribute.SetName(i_key);
            if self.point_attributes[i_key].shape[1] == 1:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple1(i_attribute)
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetScalars(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 2:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple2(i_attribute[0], i_attribute[1])
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 3:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple3(i_attribute[0], i_attribute[1], i_attribute[2])
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            else:
                if self.warning:
                    print('Check attribute dimension, only support 1D, 2D, and 3D now')

        vtkPolyData.Modified()
        self.vtkPolyData = vtkPolyData


    def landmark_transform(self, vtk_matrix):
        Trans = vtk.vtkTransform()
        Trans.SetMatrix(vtk_matrix)

        TransFilter = vtk.vtkTransformPolyDataFilter()
        TransFilter.SetTransform(Trans)
        TransFilter.SetInputData(self.vtkPolyData)
        TransFilter.Update()

        self.vtkPolyData = TransFilter.GetOutput()
        self.get_landmark_data_from_vtkPolyData()


    def landmark_reflection(self, ref_axis='x'):
        RefFilter = vtk.vtkReflectionFilter()
        if ref_axis == 'x':
            RefFilter.SetPlaneToX()
        elif ref_axis == 'y':
            RefFilter.SetPlaneToY()
        elif ref_axis == 'z':
            RefFilter.SetPlaneToZ()
        else:
            if self.warning:
                print('Invalid ref_axis!')

        RefFilter.CopyInputOff()
        RefFilter.SetInputData(self.vtkPolyData)
        RefFilter.Update()

        self.vtkPolyData = RefFilter.GetOutput()
        self.get_landmark_data_from_vtkPolyData()


    def to_vtp(self, vtp_filename):
        self.update_vtkPolyData()

        if vtk.VTK_MAJOR_VERSION <= 5:
            self.vtkPolyData.Update()

        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName("{0}".format(vtp_filename));
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(self.vtkPolyData)
        else:
            writer.SetInputData(self.vtkPolyData)
        writer.Write()


    def to_fcsv(self, fcsv_filename, landmark_name_list):
        with open(fcsv_filename, 'w') as file:
            file.write('# Markups fiducial file version = 4.10\n')
            file.write('# CoordinateSystem = 0\n')
            file.write('# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n')
            for i in range(self.points.shape[0]):
                file.write('vtkMRMLMarkupsFiducialNode_{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},,{12}\n'.format(i,
                                                                                                                        self.points[i, 0],
                                                                                                                        self.points[i, 1],
                                                                                                                        self.points[i, 2],
                                                                                                                        0.0,
                                                                                                                        0.0,
                                                                                                                        0.0,
                                                                                                                        1.0,
                                                                                                                        1,
                                                                                                                        1,
                                                                                                                        1,
                                                                                                                        landmark_name_list[i],
                                                                                                                        'vtkMRMLModelNode4'))


# if __name__ == '__main__':

#    # create a new set of landmarks by loading a VTP file
#    landmark = Easy_Landmark('A0_Sample_1_10_landmarks.vtp')
#    landmark.to_vtp('example_ld.vtp')
#
#    # create a new set of landmarks by giving a numpy array
#    landmark2 = Easy_Landmark()
#    landmark2.points = np.array([[3, 10, 2], [0, 0, 5]])
#    landmark2.to_vtp('example_ld2.vtp')
#
#    # transform a set of landmarks
#    matrix = GetVTKTransformationMatrix()
#    landmark = Easy_Landmark('A0_Sample_1_10_landmarks.vtp')
#    landmark.landmark_transform(matrix)
#    landmark.to_vtp('example_ld2.vtp')
#
#    # flip landmarks
#    landmark = Easy_Landmark('Sample_1_landmarks.vtp')
#    landmark.landmark_reflection(ref_axis='x')
#    landmark.to_vtp('flipped_example_1_landmarks.vtp')

    # create a new set of landmarks by loading fcsv
    # landmark = Easy_Landmark()
    # landmark.read_fcsv('Sample_01_UR1.fcsv', ['DCP', 'MCP', 'PGP', 'LGP'], check_name=False) # 0: DCP, 1: MCP, 2: PGP, 3:LGP for incisors
    # landmark.to_vtp('Sample_01_UR1_landmarks.vtp')
    # landmark.to_fcsv('Sample_01_UR1_tmp.fcsv', ['DCP', 'MCP', 'LDP', 'PDP'])

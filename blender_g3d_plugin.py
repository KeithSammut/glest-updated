bl_info = {
    "name": "G3D Mesh Import/Export",
    "description": "Import/Export .g3d file (Glest 3D)",
    "author": "various, updated for Blender 5.x by Bobo",
    "version": (0, 12, 1),
    "blender": (5, 1, 0),
    "location": "File > Import/Export > Glest 3D (.g3d)",
    "category": "Import-Export",
}

import bpy
from bpy.props import StringProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
import struct, os, traceback, bmesh, subprocess
from mathutils import Matrix
from math import radians

###########################################################################
# --- Data Structures ---
###########################################################################
def unpack_list(list_of_tuples):
    l = []
    for t in list_of_tuples:
        l.extend(t)
    return l


class G3DHeader:
    binary_format = "<3cB"
    def __init__(self, fileID):
        temp = fileID.read(struct.calcsize(self.binary_format))
        data = struct.unpack(self.binary_format, temp)
        self.id = (data[0] + data[1] + data[2]).decode("utf-8")
        self.version = data[3]


class G3DModelHeaderv4:
    binary_format = "<HB"
    def __init__(self, fileID):
        temp = fileID.read(struct.calcsize(self.binary_format))
        data = struct.unpack(self.binary_format, temp)
        self.meshcount = data[0]
        self.mtype = data[1]


class G3DMeshHeaderv4:
    binary_format = "<64c3I8f2I"
    texname_format = "<64c"

    def _readtexname(self, fileID):
        temp = fileID.read(struct.calcsize(self.texname_format))
        data = struct.unpack(self.texname_format, temp)
        return "".join([c.decode("ascii") for c in data if c != b'\x00'])

    def __init__(self, fileID):
        temp = fileID.read(struct.calcsize(self.binary_format))
        data = struct.unpack(self.binary_format, temp)
        self.meshname = "".join([c.decode("ascii") for c in data[:64] if c != b'\x00'])
        self.framecount = data[64]
        self.vertexcount = data[65]
        self.indexcount = data[66]
        self.diffusecolor = data[67:70]
        self.specularcolor = data[70:73]
        self.specularpower = data[73]
        self.opacity = data[74]
        self.properties = data[75]
        self.textures = data[76]

        self.customalpha = bool(self.properties & 1)
        self.istwosided = bool(self.properties & 2)
        self.noselect = bool(self.properties & 4)
        self.glow = bool(self.properties & 8)
        self.teamcoloralpha = 255 - (self.properties >> 24)

        self.hastexture = False
        self.diffusetexture = None
        self.speculartexture = None
        self.normaltexture = None

        if self.textures:
            if self.textures & 1:
                self.diffusetexture = self._readtexname(fileID)
            if self.textures & 2:
                self.speculartexture = self._readtexname(fileID)
            if self.textures & 4:
                self.normaltexture = self._readtexname(fileID)
            self.hastexture = True


class G3DMeshdataV4:
    def __init__(self, fileID, header):
        vertex_format = "<%if" % int(header.framecount * header.vertexcount * 3)
        normals_format = "<%if" % int(header.framecount * header.vertexcount * 3)
        texturecoords_format = "<%if" % int(header.vertexcount * 2)
        indices_format = "<%iI" % int(header.indexcount)

        self.vertices = struct.unpack(vertex_format, fileID.read(struct.calcsize(vertex_format)))
        self.normals = struct.unpack(normals_format, fileID.read(struct.calcsize(normals_format)))
        if header.hastexture:
            self.texturecoords = struct.unpack(texturecoords_format, fileID.read(struct.calcsize(texturecoords_format)))
        self.indices = struct.unpack(indices_format, fileID.read(struct.calcsize(indices_format)))

###########################################################################
# --- Core Mesh Importer ---
###########################################################################
def createMesh(filename, header, data, toblender, operator):
    mesh = bpy.data.meshes.new(header.meshname)
    meshobj = bpy.data.objects.new(header.meshname + "_Object", mesh)
    bpy.context.collection.objects.link(meshobj)
    bpy.context.view_layer.update()

    vertsCO = [(data.vertices[i], data.vertices[i+1], data.vertices[i+2])
               for i in range(0, header.vertexcount * 3, 3)]
    faces = [(data.indices[i], data.indices[i+1], data.indices[i+2])
             for i in range(0, len(data.indices), 3)]

    mesh.from_pydata(vertsCO, [], faces)
    mesh.update()

    if toblender:
        meshobj.rotation_euler = (radians(90), 0, 0)

    mesh.update()
    mesh.validate(clean_customdata=True)

###########################################################################
# --- Import Logic ---
###########################################################################
def G3DLoader(filepath, toblender, operator):
    print(f"\nImporting: {filepath}")
    fileID = open(filepath, "rb")
    header = G3DHeader(fileID)

    if header.id != "G3D" or header.version != 4:
        operator.report({'ERROR'}, "Unsupported or invalid G3D file")
        return {'CANCELLED'}

    modelheader = G3DModelHeaderv4(fileID)
    for _ in range(modelheader.meshcount):
        meshheader = G3DMeshHeaderv4(fileID)
        meshdata = G3DMeshdataV4(fileID, meshheader)
        createMesh(filepath, meshheader, meshdata, toblender, operator)

    fileID.close()
    print("Import finished successfully!")
    return {'FINISHED'}

###########################################################################
# --- Blender UI / Operators ---
###########################################################################
class ImportG3D(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.g3d"
    bl_label = "Import G3D"
    filename_ext = ".g3d"
    filter_glob: StringProperty(default="*.g3d", options={'HIDDEN'})
    toblender: BoolProperty(name="Rotate to Blender Orientation", default=True)

    def execute(self, context):
        try:
            return G3DLoader(self.filepath, self.toblender, self)
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


def menu_func_import(self, context):
    self.layout.operator(ImportG3D.bl_idname, text="Glest 3D Model (.g3d)")


def register():
    bpy.utils.register_class(ImportG3D)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(ImportG3D)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()

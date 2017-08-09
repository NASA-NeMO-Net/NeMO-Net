########################################################################
#
# File:   cvk2.py
# Author: Matt Zucker
# Date:   January, 2012 (Updated January, 2016)
#
# Written for ENGR 27 - Computer Vision
#
########################################################################

"""cvk2.py - OpenCV utility Kit for Python, written by Matt Zucker"""

import cv2
import numpy
import math

# Define LINE_AA for compatibility with both OpenCV 2.4 (which exports
# cv2.CV_AA) and OpenCV 3.x (which exports cv2.LINE_AA)
LINE_AA = 16

def fixKeyCode(code):
    """Some versions of OpenCV after 3.0 have waitKey return an unsigned
    int, previous verions return a signed int. This attempts to
    restore the old behavior.

    """
    return numpy.uint8(code).view(numpy.int8)

def array2cv_int(a):
    """utility function to convert a numpy array to a tuple of
    integers suitable for passing to various OpenCV functions."""
    return tuple(a.astype(int).flatten())

######################################################################

def getcontourinfo(c):
    """compute moments and derived quantities such as mean, area, and
    basis vectors from a contour as returned by cv2.findContours. """

    m = cv2.moments(c)

    s00 = m['m00']
    s10 = m['m10']
    s01 = m['m01']
    c20 = m['mu20']
    c11 = m['mu11']
    c02 = m['mu02']

    try:

        mx = s10 / s00
        my = s01 / s00

        A = numpy.array( [
                [ c20 / s00 , c11 / s00 ],
                [ c11 / s00 , c02 / s00 ] 
                ] )

        W, U, Vt = cv2.SVDecomp(A)

        ul = math.sqrt(W[0,0])
        vl = math.sqrt(W[1,0])

        ux = ul * U[0, 0]
        uy = ul * U[1, 0]

        vx = vl * U[0, 1]
        vy = vl * U[1, 1]

        mean = numpy.array([mx, my])
        uvec = numpy.array([ux, uy])
        vvec = numpy.array([vx, vy])

    except:

        mean = c[0].astype('float')
        uvec = numpy.array([1.0, 0.0])
        vvec = numpy.array([0.0, 1.0])

    return {'moments': m, 
            'area': s00, 
            'mean': mean,
            'b1': uvec,
            'b2': vvec}


######################################################################

def fetchimage(source):
    """fetches an image from a numpy.ndarray, cv2.VideoCapture, or a
    user-supplied function. """
    if isinstance(source, numpy.ndarray):
        return source
    elif str(type(source)) == "<type 'cv2.VideoCapture'>":
        ok, frame = source.read()
        return frame
    else:
        return source()

######################################################################

def getccolors():
    """returns a list of RGB colors useful for drawing segmentations
    of binary images with cv.DrawContours"""
    ccolors = [
        (  0, 255, 255),
        (255,   0,   0),
        (  0, 191, 255),
        (255,  63,   0),
        (  0, 127, 255),
        (255, 127,   0),
        (  0,  63, 255),
        (255, 191,   0),
        (  0,   0, 255),
        (255, 255,   0),
        ( 63,   0, 255),
        (191, 255,   0),
        (127,   0, 255),
        (127, 255,   0),
        (191,   0, 255),
        ( 63, 255,   0),
        (255,   0, 255),
        (  0, 255,   0),
        (255,   0, 191),
        (  0, 255,  63),
        (255,   0, 127),
        (  0, 255, 127),
        (255,   0,  63),
        (  0, 255, 191)
        ]
    return ccolors
    
######################################################################

class Widget:
    """base class for cvk2 widgets. 

You will likely want to use one of its subclasses (such as
MultiPointWidget or RectWidget) instead of instantiating an instance
of this class.

Widget objects have an array of points, self.points, which holds a
number of points that are drawn on top of an image and can be
manipulated by the user.  The array is of shape (n, 1, 2) where n
is the number of points.

To use a Widget object, call the start() method (see below).  The
Widget takes over the window until the user finishes interacting.
"""

    def __init__(self):
        """initialize this widget, sets self.points to empty (0-by-1-by-2) array"""
        self.points = numpy.empty( (0, 1, 2), dtype='float32' )
        self.imageBuffer = None
        self.active = False
        self.result = False
        self.dragging = False
        self.currentPoint = None
        self.drawMarkers = True
        self.drawLabels = False
        self.markerType = 'square'
        self.markerSize = 3
        self.clickTol = 10
        self.baseColor = (255, 0, 0)
        self.currentColor = (255, 255, 0)
        self.statusText = 'Press ENTER when finished'
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontSize = 0.4
        self.fontLineWidth = 1
        self.fontLineStyle = LINE_AA

    def start(self, window, source, keyDelay=5):
        """take over the named window provided to collect user input.

        This function enters a loop where the source provided (either
        an OpenCV image, OpenCV matrix, OpenCV VideoCapture object, or
        a function returning an OpenCV image or matrix) is queried for
        the current image, and the points are drawn on top. The user
        can drag the points around atop of the source image until
        either ENTER or ESC is pressed.

        The return value of this function is either True or False,
        depending on whether ENTER or ESC was pressed."""

        self.dragging = False
        self.active = True
        self.result = False

        cv2.setMouseCallback(window, self.mouseEvent, self)
        
        while self.active:

            srcImage = fetchimage(source)

            if (self.imageBuffer is None):
                self.imageBuffer = numpy.empty_like(srcImage)

            self.imageBuffer[:] = srcImage

            self.drawOnto(self.imageBuffer)

            cv2.imshow(window, self.imageBuffer)

            k = fixKeyCode(cv2.waitKey(keyDelay))
            
            if k >= 0:
                self.keyEvent(k)

        cv2.setMouseCallback(window, lambda e,x,y,f,p: None, None)

        self.imageBuffer[:] = srcImage
        self.drawOnto(self.imageBuffer, showCurrent=False)
        cv2.imshow(window, self.imageBuffer)
        cv2.waitKey(1)

        return self.result

    def isActive(self):
        """returns true if the Widget is currently executing the
        start() method"""
        return self.active

    def closestPoint(self, p, dmax=10):
        if not len(self.points):
            return None
        """returns the closest point in self.points to p, or None if
        the distance is greater than dmax."""
        n = len(self.points)
        pp = numpy.tile(numpy.array(p).flatten(), (n, 1, 1))
        dist = numpy.sqrt(((self.points - pp) ** 2).sum(axis=2))
        idx = numpy.argmin(dist)
        if (dist[idx] <= dmax):
            return idx
        else:
            return None

    def finish(self, result=False):
        """exits out of the start() method, generally in response to
        key press.  The result parameter is the value returned by start()"""
        self.active = False
        self.result = result

    def drawOnto(self, image, showCurrent=True):
        """draws this widget onto the given image, including all
        points and other decoration"""

        w = image.shape[1]
        h = image.shape[0]

        cv2.rectangle(image, (0,0), (w,h), self.baseColor, 4)

        cv2.putText(image, self.statusText, (6, h-8), 
                    self.fontFace, self.fontSize, self.baseColor, 
                    self.fontLineWidth, self.fontLineStyle)

        for i in range(len(self.points)):
            color = None
            if showCurrent and i == self.currentPoint:
                color = self.currentColor
            else:
                color = self.baseColor
            if self.drawMarkers:
                self.drawMarker(image, self.points[i], color)
            if self.drawLabels:
                cv2.putText(image, str(i+1), 
                            array2cv_int(self.points[i].flatten() + [5, 0]),
                            self.fontFace, self.fontSize, color,
                            self.fontLineWidth, self.fontLineStyle)

    def mouseEvent(self, event, x, y, flags, param):
        """mouse event handler installed with cv2.setMouseCallback.

        This handles the mouse input for the Widget, and will
        generally be extended in subclasses.  The base implementation
        handles dragging of existing points.

        The event handler should return True if the given event was
        handled, and False otherwise; that way, subclasses only need
        to handle events not handled by their superclass."""

        p = (x,y)

        if (self.dragging and self.currentPoint >= len(self.points)):
            self.dragging = False

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.closestPoint(p, self.clickTol)
            if (idx in range(len(self.points))):
                self.dragging = True
                self.currentPoint = idx
                return True
            else:
                return False
        elif self.dragging and event == cv2.EVENT_MOUSEMOVE:
            self.pointMoved(self.currentPoint, p, flags)
            return True
        elif self.dragging and event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            return True
        else:
            return False
    
    def keyEvent(self, k):
        """key event handler called by start() whenever cv2.waitKey()
        returns non-negative.

        The base implementation only handles the ENTER and ESC keys.

        The event handler should return True if the given event was
        handled, and False otherwise; that way, subclasses only need
        to handle events not handled by their superclass."""
        k = k % 0x100
        if k == ord('\n') or k == ord('\r'):
            self.finish(True)
            return True
        elif k == 27:
            self.finish(False)
            return False
        else:
            return False

    def pointMoved(self, index, point, flags):
        """called by mouseEvent() whenever a point is moved by the user.
        
        The base implementation simply updates the point in the
        self.points list; subclasses may use this to implement
        additional constraints on points."""
        self.points[index] = point

    def drawMarker(self, image, point, color):
        """draws a point marker onto the given image with the given color."""
        r = self.markerSize
        lt = 8
        if self.markerType == 'circle' or self.markerType == 'o':
            cv2.circle(image, array2cv_int(point), r, color, 1, lt)
        elif self.markerType == 'x':
            cv2.line(image, 
                     array2cv_int(point + (r, r)),
                     array2cv_int(point - (r, r)),
                     color, 1, lt)
            cv2.line(image, 
                     array2cv_int(point + (r, -r)),
                     array2cv_int(point + (-r, r)),
                     color, 1, lt)
        elif self.markerType == 'cross' or self.markerType == '+':
            cv2.line(image, 
                     array2cv_int(point + (r, 0)),
                     array2cv_int(point - (r, 0)),
                     color, 1, lt)
            cv2.line(image, 
                     array2cv_int(point + (0, r)),
                     array2cv_int(point - (0, r)),
                     color, 1, lt)
        else:
            delta = (r, r)
            cv2.rectangle(image, 
                          array2cv_int(point - (r, r)),
                          array2cv_int(point + (r, r)),
                          color, 1, lt)

    def save(self, filename):
        """saves the data stored by this Widget into the given file.
        
        The base class saves the self.points list; subclasses may
        override this."""
        #cv.Save(filename, self.pointsMatrix(), 'points')
        p = numpy.reshape(self.points, (len(self.points), 2))
        numpy.savetxt(filename, p, fmt="%.15G") 
        
    def load(self, filename):
        """loads the data for this Widget from the given file.
        
        The base class loads the self.points list; subclasses may
        override this."""
        #try:
        #    f = open(filename, 'r')
        #    f.close()
        #    m = cv.Load(filename, cv.CreateMemStorage(), 'points')
        #    self.setPointsFromMatrix(m)
        #    return True
        #except:
        #    return False
        try:
            data = numpy.genfromtxt(filename, dtype='float32')
            if data.shape[1] != 2:
                return False
            self.points = numpy.reshape(data, (len(data), 1, 2))
            return True
        except:
            return False
            

    def testInside(self, p, measureDist=False):
        """tests to see if the point is inside the region described by
        this Widget, or returns an approximate distance, depending on
        whether measureDist is True."""
        return cv2.pointPolygonTest(self.points, tuple(p.flatten),
                                    measureDist)

    def drawMask(self, mask, color=(255,255,255), lineType=8):
        """draw a mask corresponding to the region described by this Widget"""
        cv2.fillPoly(mask, [self.points.astype('int32')], color, lineType, 0)

######################################################################

class MultiPointWidget(Widget):

    def __init__(self, type='points'):
        """create a MultiPointWidget.
        
        The type parameter should be one of:
          - 'points': a list of disconnected points
          - 'polyline': a set of connected line segments
          - 'polygon': a closed polygon

        You can also set the following boolean flags (all True by default):
          - self.allowCreate: allows user to create points via right-click
          - self.allowDelete: allows user to delete points with Backspace key
          - self.allowReorder: allows user to reorder points with +/- keys
        """
        Widget.__init__(self)
        self.type = type
        self.drawLabels = (self.type == 'points')
        self.allowCreate = True
        self.allowDelete = True
        self.allowReorder = True
        self.statusText = 'Right-click to add points,'\
            '+ or - to reorder, ENTER when finished.'

    def drawOnto(self, image, showCurrent=True):
        """extends Widget.drawOnto() by drawing polygon/polyline"""
        if (len(self.points) >= 2 and self.type != 'points'):
            closed = (self.type == 'polygon')
            pp = self.points.astype('int32')
            cv2.polylines(image, [pp], closed, 
                          self.baseColor, 1, LINE_AA)
        Widget.drawOnto(self, image, showCurrent)

    def mouseEvent(self, event, x, y, flags, param):
        """extends Widget.mouseEvent() by allowing creation of points
        via right-click"""
        p = (x, y)
        if (self.allowCreate and event == cv2.EVENT_RBUTTONUP):
            self.currentPoint = len(self.points)
            self.points = numpy.insert(self.points, self.currentPoint, p, axis=0)
            return True
        else:
            return Widget.mouseEvent(self, event, x, y, flags, None)

    def keyEvent(self, k):
        """extends Widget.keyEvent() by supporting deletion and reordering"""
        if (Widget.keyEvent(self, k)):
            return True
        elif (k == ord('\t')):
            if (self.currentPoint in range(len(self.points))):
                self.currentPoint = self.currentPoint + 1
            else:
                self.currentPoint = 0
            return True
        elif (k == ord('-') and 
              self.allowReorder and 
              len(self.points) > 1 and
              self.currentPoint in range(len(self.points))):
            prev = self.currentPoint - 1
            if (prev < 0):
                prev = len(self.points)-1
            p = self.points[self.currentPoint].copy()
            self.points[self.currentPoint] = self.points[prev]
            self.points[prev] = p
            self.currentPoint = prev
            return True
        elif ((k == ord('+') or k== ord('=')) and
              self.allowReorder and 
              len(self.points) > 1 and
              self.currentPoint in range(len(self.points))):
            next = self.currentPoint + 1
            if (next >= len(self.points)):
                next = 0
            p = self.points[self.currentPoint].copy()
            self.points[self.currentPoint] = self.points[next]
            self.points[next] = p
            self.currentPoint = next
            return True
        elif (k == 127 and self.allowDelete and 
              self.currentPoint in range(len(self.points))):
            self.points = numpy.delete(self.points, self.currentPoint, axis=0)
            self.currentPoint = None
            return True
        else:
            return False

######################################################################

class RectWidget(Widget):
    
    def __init__(self, type='rect', allowRotate=True):
        """create a RectWidget.
        
        The type parameter should be one of:
          - 'rect': a (possibly rotated) rectangle
          - 'ellipse': a (possibly rotated) ellipse

        You can also set the self.allowRotate flag to enable or
        disable rotation."""
        Widget.__init__(self)
        self.type = type
        self.allowRotate = allowRotate
        self.angleDragging = False
        self.center = numpy.array([0,0])
        self.u = 10
        self.v = 10
        self.angle = 0
        self.nu = numpy.array([1,0])
        self.nv = numpy.array([0,1])
        self.clickCenter = numpy.array([0,0])
        self.outer = numpy.array([0, 2, 8, 6])
        
    def params(self):
        """returns a tuple (cx, cy, width, height, angle_radians)"""
        return (self.center[0], self.center[1], 
                self.u, self.v, self.angle)

    def setParams(self, params):
        """sets from a tuple (cx, cy, width, height, angle_radians)"""
        self.center = numpy.array([params[0], params[1]])
        self.u = params[2]
        self.v = params[3]
        self.angle = params[4]
        self.updatePoints()

    def initParams(self, size):
        """initialize default parameters for an image of the given
        size (height, width)"""

        w = size[1]
        h = size[0]

        self.center = numpy.array((w/2, h/2))
        self.u = w/4
        self.v = h/4
        self.angle = 0

        self.updatePoints()

    def updatePoints(self):
        """should be called to update self.points whenever the
        parameters have changed."""

        ca = math.cos(self.angle)
        sa = math.sin(self.angle)

        self.nu = numpy.array( (ca, sa)  )
        self.nv = numpy.array( (-sa, ca) ) 

        self.points = numpy.empty((0, 1, 2), dtype='float32')
        
        for dy in range(-1,2):
            for dx in range(-1,2):
                p = self.center + dx*self.nu*self.u + dy*self.nv*self.v
                self.points = numpy.insert(self.points,
                                           len(self.points),
                                           p, axis=0)

    def drawOnto(self, image, showCurrent=True):
        """extends Widget.drawOnto() by drawing rectangle/ellipse"""

        if (len(self.points) != 9):
            self.initParams(image.shape)

        if (self.type == 'ellipse'):

            cv2.ellipse(image,
                        array2cv_int(self.center),
                        (int(math.fabs(self.u)), int(math.fabs(self.v))),
                        self.angle * 180 / math.pi, 0, 360,
                        self.baseColor, 1, LINE_AA)

        else:

            cv2.polylines(image, [self.points[self.outer].astype('int32')], True, 
                         self.baseColor, 1, LINE_AA)
        
        Widget.drawOnto(self, image, showCurrent)

    def mouseEvent(self, event, x, y, flags, param):
        """extends Widget.mouseEvent() by supporting rotation"""

        if Widget.mouseEvent(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clickCenter = self.center
            return True
        elif event == cv2.EVENT_LBUTTONDOWN:
            diff = -(self.center - (x,y))
            uc = numpy.dot(diff, self.nu)
            vc = numpy.dot(diff, self.nv)
            if ((math.fabs(uc) > self.u + self.clickTol or
                 math.fabs(vc) > self.v + self.clickTol) and
                self.allowRotate):
                self.angleDragging = True
                self.angleOrig = math.atan2(diff[1], diff[0]) - self.angle
                return True
            else:
                return False
        elif event == cv2.EVENT_MOUSEMOVE and self.angleDragging:
            diff = -(self.center - (x,y))
            self.angle = math.atan2(diff[1], diff[0]) - self.angleOrig
            if (flags & cv2.EVENT_FLAG_SHIFTKEY):
                self.angle -= math.fmod(self.angle + math.pi/24, 
                                        math.pi/12) - math.pi/24
            self.updatePoints()
            return True
        elif event == cv2.EVENT_LBUTTONUP and self.angleDragging:
            self.angleDragging = False
            return True
        else:
            return False

    def pointMoved(self, index, p, flags):
        """extends Widget.pointMoved() to handle constraints"""
        dx = (index % 3) - 1
        dy = (index / 3) - 1

        if (dx or dy):
            
            diff = -(self.center - p)

            uu = self.u
            vv = self.v
            
            if (dx):
                uu = dx * numpy.dot(diff, self.nu)
                
            if (dy):
                vv = dy * numpy.dot(diff, self.nv)

            if (flags & cv2.EVENT_FLAG_SHIFTKEY):
                if (math.fabs(uu) >= 1 and math.fabs(vv) >= 1):
                    self.center = self.clickCenter
                    self.u = uu
                    self.v = vv
            else:
                du = 0.5*(uu-self.u)
                dv = 0.5*(vv-self.v)
                uu = self.u + du
                vv = self.v + dv
                if (math.fabs(uu) >= 1 and math.fabs(vv) >= 1):
                    self.u = uu
                    self.v = vv
                    self.center = self.center + dx*du*self.nu + dy*dv*self.nv

            self.updatePoints()

        else:

            self.center = numpy.array(p)
            self.updatePoints()

    def save(self, filename):
        """extends Widget.save() by saving parameters instead of points"""
        numpy.savetxt(filename, self.params(), fmt="%.15G") 
        
    def load(self, filename):
        """extends Widget.load() by loading parameters instead of points"""
        try:
            data = numpy.genfromtxt(filename, unpack=True, dtype='float32')
            self.setParams(data)
            return True
        except:
            return False


    def testInside(self, p, measureDist=False):
        """extends Widget.testInside() to support distance to ellipse.
        Note: distance to ellipse is approximate."""
        if self.type == 'ellipse': 
            pdiff = -(self.center-p)
            uc = numpy.dot(pdiff, nu) / self.u
            vc = numpy.dot(pdiff, nv) / self.v
            d2 = uc*uc + vc*vc
            if measureDist:
                d = math.sqrt(d2)
                if d:
                    uc /= d
                    vc /= d
                else:
                    uc = 1
                    vc = 0
                pc = self.center + uc*self.u*self.nu + vc*self.v*self.nv
                diff = -(pc - p)
                dc = math.sqrt(numpy.dot(diff, diff))
                if d2 < 1:
                    return dc
                else:
                    return -dc
            elif d2 == 1:
                return 0
            elif d2 < 1:
                return 1
            elif d2 > 1:
                return -1
        else:
            if len(self.points) != 9:
                updatePoints()
            ppoly = self.points[self.outer]
            return cv2.pointPolygonTest( ppoly, tuple(p.flatten()),
                                         measureDist )

    def drawMask(self, mask, color=(255,255,255), lineType=8):
        """extends Widget.drawMask() to support drawing ellipse."""
        if self.type == 'ellipse':
            cv2.ellipse(mask, 
                        array2cv_int(self.center), 
                        (int(math.fabs(self.u)), int(math.fabs(self.v))),
                        self.angle * 180 / math.pi, 0, 360,
                        color, -1, lineType)
        else:
            ppoly = self.points[self.outer]
            cv2.fillPoly(mask, [ppoly.astype('int32')], color, lineType, 0)

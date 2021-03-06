"""
http://blog.christianperone.com/wp-content/uploads/2009/02/modis.py

This is a script to download modis images and shapefiles. It will plot using
Matplotlib and the Basemap toolkit, so, you must install it first.

Matplotlib
  http://matplotlib.sourceforge.net/

Basemap Toolking
  http://matplotlib.sourceforge.net/basemap/doc/html/users/installing.html

- Christian S. Perone
http://pyevolve.sourceforge.net/wordpress
"""

print "Loading modules... ",
import dbflib
from mpl_toolkits.basemap import Basemap
import pylab as p
from PIL import Image
import os, os.path, sys
import tempfile, zipfile
import urllib2, time
print "done!"

# ======== Change here =========
# Satellite name
# "aqua" or "terra"
MODIS_SATNAME = "terra"

# Subset name
# See the list on http://rapidfire.sci.gsfc.nasa.gov/subsets
SUBSET_NAME = "FAS_Brazil5"

# The pixel size
# "2km", "1km", "500m" or "250m"
RAPIDFIRE_RES = "500m"

# The active fire/hotspot for the lasts time
# "24h", "48h", "7d"
FIRE_LASTS = "24h"
# ==============================

# Year is in format "yyyy"
SUBSET_YEAR = str(time.localtime()[0])
# The day of the year
SUBSET_CODE = "041" #"%03d" % (time.localtime()[7])
TUPLE_MODIS = (SUBSET_NAME, SUBSET_YEAR, SUBSET_CODE, MODIS_SATNAME, RAPIDFIRE_RES)
# The download urls
URL_RAPIDFIRE_SUBSET = "http://rapidfire.sci.gsfc.nasa.gov/subsets/?subset=%s.%s%s.%s.%s.zip" % TUPLE_MODIS
URL_METADATA         = "http://rapidfire.sci.gsfc.nasa.gov/subsets/?subset=%s.%s%s.%s.%s.txt" % TUPLE_MODIS
URL_FIRE_SHAPES      = "http://firefly.geog.umd.edu/shapes/zips/Global_%s.zip" % FIRE_LASTS
IMAGE_FILE           = '%s.%s%s.%s.%s.jpg' % TUPLE_MODIS

def parseTerm(metadata, term):
   """ Parses the txt or html metadata file """
   start = metadata.find(term + ":") + len(term)+1
   end   = metadata[start:].find("\n")
   val   = float(metadata[start:start+end])
   return val

def downloadString(url):
   """ Returns a string with the url contents """
   filein = urllib2.urlopen(url)
   data   = filein.read()
   filein.close()
   return data

def download(url, fout):
   """ Saves the url file to fout filename """
   filein  = urllib2.urlopen(url)
   fileout = open(fout, "wb")

   while True:
      bytes = filein.read(1024)
      fileout.write(bytes)

      if bytes == "": break

   filein.close()
   fileout.close()

def run_main():
   """ Main """
   print "Downloading last near real-time true-color image from MODIS... ",
   download(URL_RAPIDFIRE_SUBSET, "data.zip")

   try:
      zipf    = zipfile.ZipFile('data.zip')
   except zipfile.BadZipfile:
      print "\n\n\tError: BadZipfile, maybe the data is not yet ready on MODIS site !\n"
      sys.exit(-1)

   tempdir = tempfile.mkdtemp()

   for name in zipf.namelist():
      data    = zipf.read(name)
      outfile = os.path.join(tempdir, name)
      f       = open(outfile, 'wb')
      f.write(data)
      f.close()

   zipf.close()

   image_path  = os.path.join(tempdir, IMAGE_FILE)
   image_modis = Image.open(image_path)
   print "done !"

   print "Downloading MODIS image metadata... ",
   metadata = downloadString(URL_METADATA)
   ll_lon = parseTerm(metadata, "LL lon")
   ll_lat = parseTerm(metadata, "LL lat")
   ur_lon = parseTerm(metadata, "UR lon")
   ur_lat = parseTerm(metadata, "UR lat")
   print "done !"

   print "Downloading shape files from MODIS rapid fire... ",
   download(URL_FIRE_SHAPES, "shapes.zip")
   zipf = zipfile.ZipFile('shapes.zip')

   for name in zipf.namelist():
      data    = zipf.read(name)
      outfile = os.path.join(tempdir, name)
      f       = open(outfile, 'wb')
      f.write(data)
      f.close()
   zipf.close()
   print "done !"

   print "Parsing shapefile... ",

   shape_path = os.path.join(tempdir, 'Global_%s' % FIRE_LASTS)
   dbf        = dbflib.open(shape_path)
   rec_count  = dbf.record_count()

   xlist      = [dbf.read_record(i)['LONGITUDE'] for i in xrange(rec_count)]
   ylist      = [dbf.read_record(i)['LATITUDE'] for i in xrange(rec_count)]
   confidence = [dbf.read_record(i)['CONFIDENCE'] for i in xrange(rec_count)]
   dbf.close()
   print "%d records read !" % (rec_count,)

   print "Drawing map... ",
   m = Basemap(projection='cyl', llcrnrlat=ll_lat, urcrnrlat=ur_lat,\
               llcrnrlon=ll_lon, urcrnrlon=ur_lon, resolution='h')
   m.drawcoastlines()
   m.drawmapboundary(fill_color='aqua')
   m.scatter(xlist, ylist, 20, c=confidence, cmap=p.cm.hot, marker='o', edgecolors='none', zorder=10)
   m.imshow(image_modis)
   print "done !"

   os.remove("data.zip")
   os.remove("shapes.zip")

   for file in os.listdir(tempdir):
     os.unlink(os.path.join(tempdir, file))
   os.rmdir(tempdir)

   p.title("The recent fire hotspots for lasts %s, pixel size of %s" % (FIRE_LASTS, RAPIDFIRE_RES))
   p.show()

if __name__ == "__main__":
   run_main()
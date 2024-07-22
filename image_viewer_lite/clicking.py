#
# An example demonstrating interactive clicking
#
# Recommend to use an interactive iPython environment
# Or, in command line, do $ ipython -i clicking.py
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import Angle
import astropy.units as u

# magic command that ensures interactive in notebooks
# %matplotlib notebook

"""
data_cube: numpy.ndarray
    2D map: (y, x)
    3D cube: (z, y, x)
    
zscale:
    'minmax': minmax for everything
    else: determined by initial slice

logscale: bool
    True: adjust the scales for colorbar & profiles in log; blank data <= 0
    False: linear scale for all

header: astropy.Header object

--
Future updates:
- auto fits reading + directly from fits (data_cube format, x/y/z units, WCS...)
- hide "physical" stuff when header is None
- "drag" the pointer: add to "clicking"
- aperture: at least circular
- zoom
- better zscale (adjust every click? some outlier rejection?)
- logscale --> interactive
- alternative side panels: stats panel, histogram, aperture z-profile......
  - make each panel as an object? (method: initilize, update)

"""
class PointMover:
    def __init__(self, data_cube, zscale='minmax', logscale=False,
                 header=None):
        # General definitions
        self.color_x = 'tab:green'
        self.color_y = 'tab:orange'
        self.color_z = 'tab:blue'
        
        # Initilize a clicking position at the spatial center
        self.naxis = len(data_cube.shape)
        if self.naxis == 2:
          self.cur_2d_map = data_cube
          self.naxis_x = data_cube.shape[1]
          self.naxis_y = data_cube.shape[0]
          self.naxis_z = 0
        elif self.naxis == 3:
          self.cur_2d_map = data_cube[0]
          self.naxis_x = data_cube.shape[2]
          self.naxis_y = data_cube.shape[1]
          self.naxis_z = data_cube.shape[0]
        else:
          raise ValueError(f'data_cube must be 2D or 3D. The input data_cube.shape = {self.naxis}.')
        self.cur_x = int(self.naxis_x // 2)
        self.cur_y = int(self.naxis_y // 2)
        self.cur_z = 0
        self.data_cube = data_cube

        # log scale
        self.logscale = logscale
        if logscale:
            self.data_cube[self.data_cube <= 0] = np.nan

        # zscale
        if zscale == 'minmax':
            self.lim = [np.nanmin(data_cube), np.nanmax(data_cube)]
        else:
            self.lim = [None, None]

        # units (for labels)
        # no plan for x/y units (well...Delta arcsec...) right now.
        self.unit = 'arb. unit'
        self.zunit = 'arb. unit'
        self.wcs = None
        if header is not None:
            self.header_handler(header)
        
        # Initialize fig and axs
        plt.style.use('dark_background')

        self.fig = plt.figure(figsize=(12.8, 7.6))
        if self.naxis == 3:
            self.gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[2, 1], height_ratios=[1, 1, 1])
            self.ax_3 = self.fig.add_subplot(self.gs[2, 1])
        elif self.naxis == 2:
            self.gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[2, 1], height_ratios=[1, 1])
        self.ax_main = self.fig.add_subplot(self.gs[:, 0])
        self.ax_1 = self.fig.add_subplot(self.gs[0, 1])
        self.ax_2 = self.fig.add_subplot(self.gs[1, 1])

        # Initilize the display: title
        self.reset_title()
        # Initilize the display: main ax
        norm = LogNorm(vmin=self.lim[0], vmax=self.lim[1]) if logscale else Normalize(vmin=self.lim[0], vmax=self.lim[1])
        self.im = self.ax_main.imshow(self.cur_2d_map, origin='lower', cmap='inferno', norm=norm)
        cbar = plt.colorbar(self.im, ax=self.ax_main, location='bottom')
        cbar.set_label(self.unit)
        self.point, = self.ax_main.plot([self.cur_x], [self.cur_y], color=self.color_z, marker='o', markersize=10)
        ## WCS: main ax
        if self.wcs is not None:
            # RA
            xticks = self.ax_main.get_xticks()
            RA, Dec = self.wcs.wcs_pix2world(xticks, [self.cur_y] * len(xticks), 0)
            xlabels = Angle(RA * u.deg).to_string(u.hour, precision=1)
            self.ax_main.set_xticks(xticks)
            self.ax_main.set_xticklabels(xlabels, rotation=30)
            self.ax_main.set_xlabel('RA', size=15)
            # Dec
            yticks = self.ax_main.get_yticks()
            RA, Dec = self.wcs.wcs_pix2world([self.cur_x] * len(yticks), yticks, 0)
            ylabels = Angle(Dec * u.deg).to_string(u.deg, precision=1)
            self.ax_main.set_yticks(yticks)
            self.ax_main.set_yticklabels(ylabels)
            self.ax_main.set_ylabel('Dec', size=15)

        # Initilize the display: profiles
        self.plot_profiles()        

        # Spatial: Link the point to clicking event
        self.cid = \
            self.point.figure.canvas.mpl_connect('button_press_event', self)
        # Spectral: Create slidebar for z
        if self.naxis == 3:
            slider_ax = plt.axes([0.1, 0.01, 0.5, 0.02], facecolor='cyan')  # lightgoldenrodyellow
            self.slider_z = Slider(slider_ax, 'z slice \n [image]', 0, self.naxis_z - 1, valinit=self.cur_z, valstep=1)
            self.slider_z.on_changed(self.update_slice)

        # Initial draw
        self.fig.suptitle('Image Viewer Lite', size=24)
        self.fig.tight_layout()
        plt.draw()
        plt.show()

    def header_handler(self, header):
        # self.unit
        self.unit = 'unidentified_keyword'
        unit_keywords = {'BUNIT'}
        for kw in unit_keywords:
            if kw in header:
                self.unit = header[kw].strip()
                break
        # self.zunit
        self.zunit = 'unidentified_keyword'
        zunit_keywords = {'CUNIT3'}
        for kw in zunit_keywords:
            if kw in header:
                self.zunit = header[kw].strip()
                break
        # wcs stuff
        self.wcs = WCS(header, naxis=2)
        # spectral stuff
        self.crval3 = header['CRVAL3'] if 'CRVAL3' in header else None
        self.cdelt3 = header['CDELT3'] if 'CDELT3' in header else None
        self.crpix3 = header['CRPIX3'] if 'CRPIX3' in header else None

    def __call__(self, event):
        #
        # Grab clicking event information
        #
        print('click', event)
        if event.inaxes != self.point.axes:
            return
        self.cur_x, self.cur_y = int(round(event.xdata)), int(round(event.ydata))
        #
        # Update pointer position in the first axis
        #
        self.reset_title()
        self.point.set_data(self.cur_x, self.cur_y)
        self.point.figure.canvas.draw()
        """
        # From old code. Probably not necessary. clear() moved to plot_profiles()
        # Clear the other axes. Avoid nan / inf problem here if you want.
        self.ax[1].clear()
        if not np.isfinite(self.cur_2d_map[i, j]):
            return
        """
        # Update the profiles (or other possible side-panels)
        self.update_profiles()
        self.fig.canvas.draw()

    def update_slice(self, val):
        # Update the current z index
        self.cur_z = int(self.slider_z.val)
        self.cur_2d_map = self.data_cube[self.cur_z, :, :]
        # Update the main image
        self.im.set_data(self.cur_2d_map)
        # Update the profiles
        self.update_profiles()
        # Update the plot title
        self.reset_title()
        # Redraw the figure
        self.fig.canvas.draw()

    def reset_title(self):
        xlabel, ylabel, zlabel = '', '', 'N/A'
        if self.wcs is not None:
            RA, Dec = self.wcs.wcs_pix2world([self.cur_x], [self.cur_y], 0)
            xlabel = Angle(RA * u.deg).to_string(u.hour, precision=1)[0]
            ylabel = Angle(Dec * u.deg).to_string(u.deg, precision=1)[0]
            if self.cdelt3 is not None:
                zlabel = str(int((self.cur_z - self.crpix3) * self.cdelt3 + self.crval3)) + ' ' + self.zunit
        if self.naxis == 3:
            title = f'Image=({self.cur_x}, {self.cur_y}, {self.cur_z}) \n Physical=({xlabel}, {ylabel}, {zlabel})'
        elif self.naxis == 2:
            title = f'Image=({self.cur_x}, {self.cur_y}) \n Physical=({xlabel}, {ylabel})'
        self.ax_main.set_title(title)

    def plot_profiles(self):
        # x profile
        self.ax_1.clear()
        # self.ax_1.set_title('x profile')  # Might be used for other side panels. Don't name them ax_z.
        label_x = f'x profile'
        if self.naxis == 3:
          self.profile_x, = self.ax_1.plot(np.arange(self.naxis_x), self.data_cube[self.cur_z, self.cur_y, :], color=self.color_x, label=label_x)
        elif self.naxis == 2:
          self.profile_x, = self.ax_1.plot(np.arange(self.naxis_x), self.data_cube[self.cur_y, :], color=self.color_x, label=label_x)
        self.vline_x = self.ax_1.axvline(self.cur_x, color=self.color_z, alpha=0.5, linestyle=':')
        ax_1_image = self.ax_1.twiny()
        ax_1_image.set_xlim(self.ax_1.get_xlim())
        ax_1_image.set_xlabel('x: image', color=self.color_x)
        self.ax_1.set_ylim(self.lim)
        if self.logscale:
            self.ax_1.set_yscale('log')
        if self.wcs is not None:
            xticks = self.ax_1.get_xticks()
            RA, Dec = self.wcs.wcs_pix2world(xticks, [self.cur_y] * len(xticks), 0)
            xlabels = Angle(RA * u.deg).to_string(u.hour, precision=1)
            self.ax_1.set_xticks(xticks)
            self.ax_1.set_xticklabels(xlabels, rotation=30)
        self.ax_1.set_xlabel('x: physical', color=self.color_x)
        self.ax_1.set_ylabel(self.unit)
        self.ax_1.legend(loc='upper left')
        # y profile
        self.ax_2.clear()
        # self.ax_2.set_title('y profile')
        label_y = f'y profile'
        if self.naxis == 3:
          self.profile_y, = self.ax_2.plot(np.arange(self.naxis_y), self.data_cube[self.cur_z, :, self.cur_x], color=self.color_y, label=label_y)
        elif self.naxis == 2:
          self.profile_y, = self.ax_2.plot(np.arange(self.naxis_y), self.data_cube[:, self.cur_x], color=self.color_y, label=label_y)
        self.vline_y = self.ax_2.axvline(self.cur_y, color=self.color_z, alpha=0.5, linestyle=':')
        ax_2_image = self.ax_2.twiny()
        ax_2_image.set_xlim(self.ax_2.get_xlim())
        ax_2_image.set_xlabel('y: image', color=self.color_y)
        self.ax_2.set_ylim(self.lim)
        if self.logscale:
            self.ax_2.set_yscale('log')
        if self.wcs is not None:
            xticks = self.ax_2.get_xticks()
            RA, Dec = self.wcs.wcs_pix2world([self.cur_x] * len(xticks), xticks, 0)
            xlabels = Angle(Dec * u.deg).to_string(u.deg, precision=1)
            self.ax_2.set_xticks(xticks)
            self.ax_2.set_xticklabels(xlabels, rotation=30)
        self.ax_2.set_xlabel('y: physical', color=self.color_y)
        self.ax_2.set_ylabel(self.unit)
        self.ax_2.legend(loc='upper left')
        # z profile
        if self.naxis == 3:
          self.ax_3.clear()
          label_z = f'z profile'
          self.profile_z, = self.ax_3.plot(np.arange(self.naxis_z), self.data_cube[:, self.cur_y, self.cur_x], color=self.color_z, label=label_z)
          self.vline_z = self.ax_3.axvline(self.cur_z, color=self.color_z, alpha=0.5, linestyle=':')
          ax_3_image = self.ax_3.twiny()
          ax_3_image.set_xlim(self.ax_3.get_xlim())
          ax_3_image.set_xlabel('z: image', color=self.color_z)
          self.ax_3.set_ylim(self.lim)
          if self.logscale:
              self.ax_3.set_yscale('log')
          if self.zunit != 'unidentified_keyword':
              xticks = self.ax_3.get_xticks()
              vel = ((xticks - self.crpix3) * self.cdelt3 + self.crval3).astype(int)
              self.ax_3.set_xticks(xticks)
              self.ax_3.set_xticklabels(vel, rotation=30)
          self.ax_3.set_xlabel(f'z: physical [{self.zunit}]', color=self.color_z)
          self.ax_3.set_ylabel(self.unit)
          self.ax_3.legend(loc='upper left')

    def update_profiles(self):
        # x profile
        self.vline_x.set_xdata(self.cur_x)
        if self.naxis == 3:
            self.profile_x.set_ydata(self.data_cube[self.cur_z, self.cur_y, :])
        elif self.naxis == 2:
            self.profile_x.set_ydata(self.data_cube[self.cur_y, :])
        # y profile
        self.vline_y.set_xdata(self.cur_y)
        if self.naxis == 3:
            self.profile_y.set_ydata(self.data_cube[self.cur_z, :, self.cur_x])
        elif self.naxis == 2:
            self.profile_y.set_ydata(self.data_cube[:, self.cur_x])
        # z profile
        if self.naxis == 3:
            self.vline_z.set_xdata(self.cur_z)
            self.profile_z.set_ydata(self.data_cube[:, self.cur_y, self.cur_x])
        # self.fig.canvas.draw()  Do it in __call__() or slice()

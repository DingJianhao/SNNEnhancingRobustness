import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

class LargeFig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.default_size = (8.27 * 2.54, 11.69 * 2.54)
        if isinstance(kwargs['x'], str) and 'linewidth' in kwargs['x']:
            kwargs['x'] = self.default_size[0] * float(kwargs['x'].split('linewidth')[0])

        if isinstance(kwargs['y'], str) and 'pageheight' in kwargs['y']:
            kwargs['y'] = self.default_size[1] * float(kwargs['y'].split('pageheight')[0])
        self.fig_size = (kwargs['x'], kwargs['y'])
        self.cm2inch = 1/2.54
        self.inch2cm = 2.54
        self.sizes = {}

        font = {'family': 'Arial',
                'weight': 'medium',}

        matplotlib.rc('font', **font)

        self.fig = plt.figure(
            figsize=(self.fig_size[0] * self.cm2inch,
                     self.fig_size[1] * self.cm2inch)
        )

    def get_fig_size(self):
        # return plt.figure(
        #     figsize=(self.fig_size[0] * self.cm2inch,
        #              self.fig_size[1] * self.cm2inch)
        # )
        return self.fig_size[0] * self.cm2inch, self.fig_size[1] * self.cm2inch

    # def get_fig(self):
    #     self.fig = plt.figure(
    #         figsize=(self.fig_size[0] * self.cm2inch,
    #                  self.fig_size[1] * self.cm2inch)
    #     )

    # def get_axes(self, left_upper, fig_size):
    #     x, y = fig_size
    #     left_lower_x = left_upper[0] * self.cm2inch
    #     left_lower_y = (self.fig_size[1] - left_upper[1] - y) * self.cm2inch
    #     return left_lower_x / self.get_fig_size()[0], left_lower_y / self.get_fig_size()[1], \
    #            x * self.cm2inch / self.get_fig_size()[0], y * self.cm2inch / self.get_fig_size()[1]

    def add_axes(self, left_upper, fig_size, name=None, **kwargs):
        x, y = fig_size
        left_lower_x = left_upper[0] * self.cm2inch
        left_lower_y = (self.fig_size[1] - left_upper[1] - y) * self.cm2inch
        f = left_lower_x / self.get_fig_size()[0], left_lower_y / self.get_fig_size()[1], \
               x * self.cm2inch / self.get_fig_size()[0], y * self.cm2inch / self.get_fig_size()[1]
        if name is None:
            name = len(self.keys())
        self[name] = self.fig.add_axes(f, **kwargs)
        self.sizes[name] = (x, y)
        return self[name]

    def add_axes_with_demo_png(self, left_upper, fig_size, demo_png_file, fix_height=None, fix_width=None, name=None, **kwargs):
        image = Image.open(demo_png_file)
        x, y = fig_size
        if fix_width is not None and fix_width:
            x = fig_size[0]
            y = (x / image.size[0]) * image.size[1]
        if fix_height is not None and fix_height:
            y = fig_size[1]
            x = (y / image.size[1]) * image.size[0]

        left_lower_x = left_upper[0] * self.cm2inch
        left_lower_y = (self.fig_size[1] - left_upper[1] - y) * self.cm2inch
        f = left_lower_x / self.get_fig_size()[0], left_lower_y / self.get_fig_size()[1], \
               x * self.cm2inch / self.get_fig_size()[0], y * self.cm2inch / self.get_fig_size()[1]
        if name is None:
            name = len(self.keys())
        self[name] = self.fig.add_axes(f, **kwargs)
        self.sizes[name] = (x, y)
        self[name].imshow(image)
        return self[name]

    def savefig(self, *args, **kwargs):
        plt.savefig(*args, **kwargs)

    def show(self):
        plt.show()

    def axis_off(self, name):
        self[name].axis('off')

    def axis_on(self, name):
        self[name].axis('on')

    def empty(self, name):
        self[name].axis('on')
        self[name].set_xticks([])
        self[name].set_yticks([])

    def remove_top_right_spine(self, name):
        self[name].spines['right'].set_visible(False)
        self[name].spines['top'].set_visible(False)

    def remove_all_spine(self, name):
        self[name].spines['right'].set_visible(False)
        self[name].spines['top'].set_visible(False)
        self[name].spines['left'].set_visible(False)
        self[name].spines['bottom'].set_visible(False)

    def remove_axes(self, name):
        self[name].set_xticks([])
        self[name].set_yticks([])

    def change_axis_spines_width(self, name, width):
        self[name].spines['top'].set_linewidth(width)
        self[name].spines['bottom'].set_linewidth(width)
        self[name].spines['left'].set_linewidth(width)
        self[name].spines['right'].set_linewidth(width)

    def change_ticks_width(self, name, width):
        self[name].xaxis.set_tick_params(width=width)
        self[name].yaxis.set_tick_params(width=width)

    def give_fig_id(self, name, id, x=0.5, y=0.5, fontsize=14, fontname='Arial', weight='bold'):
        '''

        :param name:
        :param id:
        :param x: x from left upper (positive means right)
        :param y: y from left upper (positive means upper)
        :return:
        '''
        fig_x, fig_y = self.sizes[name]
        x_ = x / fig_x
        y_ = y / fig_y
        return self[name].text(-x_, 1 + y_, id,
             horizontalalignment='center',
             verticalalignment='center',
             transform=self[name].transAxes, fontsize=fontsize, fontname=fontname, weight=weight)
#
#
# # ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# # ax4 = fig.add_axes([0.5,0.5,0.16,0.16])
# # print(type(ax3))  # <class 'matplotlib.axes._axes.Axes'>
# # plt.show()
if __name__ == '__main__':
    # fig = plt.figure(figsize=(8.27, 11.69,))
    # ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax4 = fig.add_axes([0.5, 0.7, 0.16, 0.4])
    # ax3.plot([0, 0], [0, 1])
    # plt.show()

    largefig = LargeFig(x='1linewidth', y='0.5pageheight')
    ax1 = largefig.add_axes(left_upper=(2, 2), fig_size=(2, 2))
    largefig.empty(0)
    ax1.set_title('xxx')
    largefig.savefig('xxx.pdf')
    largefig.show()

    # import matplotlib.pyplot as plt
    # largefig = LargeFig(x=21, y=29.7)
    # fig = plt.figure(
    #     figsize=largefig.get_fig_size()
    # )
    # ax = fig.add_axes([*largefig.get_axes((2,2), (2,2))])
    # ax.plot([0,0], [0,1])
    # plt.savefig('xxx.pdf')
#
#     plt.show()
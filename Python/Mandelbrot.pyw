import wx
import time
import pyopencl as cl
from PIL import ImageFont, Image
import numpy as np
from numpy import int32, float32

USE_JULIA = False
MONOCHROME = True
OVERLAPPED = True

W = 1024
H = 768
N = (W * H)
MAX_ITERATIONS = 256
ZOOM = 1.5

RAINBOW = 0
GREEN = 1
GREEN_INVERTED = 2
BLUE = 3
BLUE_INVERTED = 4
COLOR_MODES = 5

class cl_unit(object):
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.make_kernels()

        mf = cl.mem_flags
        self.cl_mem = cl.Buffer(self.ctx, mf.WRITE_ONLY, N * 4)
        self.cl_min = cl.Buffer(self.ctx, mf.WRITE_ONLY, 4)
        self.cl_colour_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, N * 4)        
        self.max = MAX_ITERATIONS
        self.next_buffer = np.empty((N,1), dtype=np.int32)
        self.cl_copy_handle = None

        if USE_JULIA:
            self.x = -2.0
            self.y = -1.5
        else:
            self.x = -2.5
            self.y = -1.5

        self.w = 4.0
        self.h = 3.0
        self.xc = -0.8
        self.yc = 0.156
        self.elapsed = 0.
        self.color_mode = GREEN
        #[self update];

    def make_kernels(self):
        prg = cl.Program(self.ctx, """
            __kernel void min_value(global float *buffer, int count, global float* result)
            {
                int i;
                float current_min = buffer[0];
                
                for (i = 1; i < count; i++) {
                    if (buffer[i] < current_min) {
                        current_min = buffer[i];
                    }
                }
                result[0] = current_min;
            }

            void hsv_to_rgb(float *r, float *g, float *b, float h, float s, float v) {
                h /= 60;
                int i = floor(h);
                float f = h - i;
                float p = v * (1 - s);
                float q = v * (1 - s * f);
                float t = v * (1 - s * (1 - f));
                switch (i) {
                    case 0: *r = v; *g = t; *b = p; break;
                    case 1: *r = q; *g = v; *b = p; break;
                    case 2: *r = p; *g = v; *b = t; break;
                    case 3: *r = p; *g = q; *b = v; break;
                    case 4: *r = t; *g = p; *b = v; break;
                    case 5: *r = v; *g = p; *b = q; break;
                }
            }   

            __kernel void colour_green(global float *input_buffer, 
                                       global char4 *output_buffer, 
                                       global float *min_value, float max_value)
            {   
                size_t index = get_global_id(0);
                float4 rgba;
                
                if (input_buffer[index] >= max_value) {
                    rgba = (float4)(0.,0.,0.,0.);
                } else {
                    float p = log(input_buffer[index] - *min_value + 1) / log(max_value - *min_value + 1);
                    float r, g, b;
                    hsv_to_rgb(&r, &g, &b, 120.0, 1.0, 1.0 - p);
                    rgba = (r, g, b, 1.0);
                }
                char4 retval;
                retval = (char4) ((char) (255.0 * rgba.s0), (char) (255.0 * rgba.s1), (char) (255.0 * rgba.s2), (char) (255.0 * rgba.s3));

                output_buffer[index] = retval;
            }

            __kernel void julia(int _max, int _xs, int _ys,
                                float _x, float _y, 
                                float _w, float _h, 
                                float _xc, float _yc,
                                global float *output)
            {
                size_t index = get_global_id(0);
                float result;
                float i = index % _xs;
                float j = index / _xs;
                float x = _x + _w * (i / _xs);
                float y = _y + _h * (j / _ys);
                int iteration = 0;
                while (x * x + y * y < 4 && iteration < _max) {
                    float temp = x * x - y * y + _xc;
                    y = 2 * x * y + _yc;
                    x = temp;
                    iteration++;
                }
                result = iteration;
                output[index] = result;
            }

            __kernel void mandelbrot(int _max, int _xs, int _ys, float _x, float _y, float _w, float _h, global float *output)
            {
                size_t index = get_global_id(0);
                float result;
                float i = index % _xs;
                float j = index / _xs;
                float x0 = _x + _w * (i / _xs);
                float y0 = _y + _h * (j / _ys);
                float x1 = x0 + 1;
                float x4 = x0 - 1.0f / 4;
                float q = x4 * x4 + y0 * y0;
                if (q * (q + x4) * 4 < y0 * y0) {
                    result = _max;
                }
                else if ((x1 * x1 + y0 * y0) * 16 < 1) {
                    result = _max;
                }
                else {
                    float x = 0;
                    float y = 0;
                    int iteration = 0;
                    while (x * x + y * y < 4 && iteration < _max) {
                        float temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        iteration++;
                    }
                    result = iteration;
                }
                output[index] = result;
            }
        """)

        self.kernels = prg.build()    

    def update(self):
        if USE_JULIA:
            self.kernels.julia(self.queue, (N,1), None, 
                         int32(self.max), int32(W), int32(H), float32(self.x), float32(self.y), float32(self.w), 
                         float32(self.h), float32(self.xc), float32(self.yc), self.cl_mem)
        else:
            self.kernels.mandelbrot(self.queue, (N,1), None,
                         int32(self.max), int32(W), int32(H), float32(self.x), float32(self.y), float32(self.w), 
                         float32(self.h), self.cl_mem)

        if MONOCHROME:
            if OVERLAPPED:
                if self.cl_copy_handle:
                    self.cl_copy_handle.wait()

                self.current_buffer = self.next_buffer

                self.next_buffer = np.empty((N,1), dtype=np.float32)
                self.cl_copy_handle = cl.enqueue_copy(self.queue, self.next_buffer, self.cl_mem)
            else:
                self.current_buffer = np.empty((N,1), dtype=np.float32)
                cl.enqueue_copy(self.queue, self.current_buffer, self.cl_mem).wait()
            
        else:        
            self.kernels.min_value(self.queue, (1, 1), None,
                              self.cl_mem, int32(N), self.cl_min)

            self.kernels.colour_green(self.queue, (N,1), None,
                              self.cl_mem, self.cl_colour_buffer, self.cl_min, float32(self.max))
            output = np.empty((N,1), dtype=np.int32)                              
            cl.enqueue_copy(self.queue, output, self.cl_colour_buffer).wait()

        return self.current_buffer
    

class Panel(wx.Panel):
    def __init__(self, parent, cl_model, font, update_rate = 1.0):
        super(Panel, self).__init__(parent, -1)
        self.cl_model = cl_model
        self.parent = parent
        self.font = font
        self.update_rate = update_rate
        self.count = 0
        self.fps = 0.0
        self.starttime = time.clock()
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right)
        self.initial = True
        self.update()
        wx.CallLater(10,self.timer_func)

    def timer_func(self):
        self.on_left(zoom=1.01)
        #wx.CallLater(10, self.timer_func)

    def on_left(self, event = None, zoom = ZOOM):
        if event:
            point = event.GetPosition()
        else:
            if self.initial:
                point = 424,326
                self.initial = False
            else:
                point = W/2, H/2

        point = float(point[0]), H - float(point[1])

        
        #if USE_JULIA:
        #    self.cl_model.xc = 2 * float(point[0])/W - 1
        #    self.cl_model.yc = 2 * float(point[1])/H - 1
        #else:
        if True:
            x = self.cl_model.x + self.cl_model.w * (point[0] / W)
            y = self.cl_model.y + self.cl_model.h * (point[1] / H)
            self.cl_model.w /= zoom
            self.cl_model.h /= zoom
            self.cl_model.x = x - self.cl_model.w / 2.0
            self.cl_model.y = y - self.cl_model.h / 2.0
        self.update()
        
        if not event:
            wx.CallLater(0, self.on_left, zoom =1.01)
        
    def on_right(self, event = None):
        if event:
            point = event.GetPosition()
        else:
            point = W/2, H/2
        point = float(point[0]), H - float(point[1])
    
        x = self.cl_model.x + self.cl_model.w * point[0] / W
        y = self.cl_model.y + self.cl_model.h * point[1] / H
        self.cl_model.w *= ZOOM
        self.cl_model.h *= ZOOM
        self.cl_model.x = x - self.cl_model.w / 2
        self.cl_model.y = y - self.cl_model.h / 2
        self.update()
    
    def update_titlebar(self):
        self.parent.SetTitle("OpenCL Mandelbrot - simple example (%.2f fps)" % self.fps)

    def update(self):
        self.Refresh()
        self.Update()
        #wx.CallLater(1000, self.update)

    def create_bitmap(self):
        # get the opencl data
        buffer = self.cl_model.update()
        data = buffer.tostring()

        if MONOCHROME:
            # create a grayscale image
            image = Image.frombuffer("F", (W, H), data).convert('RGB')
        else:
            image = Image.frombuffer("RGB", (W, H), data)

        width, height = image.size
        data = image.tostring()
        bitmap = wx.BitmapFromBuffer(width, height, data)

        self.count += 1
        now = time.clock()
        elapsed = now - self.starttime
        if elapsed >= self.update_rate:
            self.fps = self.count / elapsed
            self.starttime = now
            self.count = 0
            self.update_titlebar()
        
        return bitmap

    def on_paint(self, event):
        bitmap = self.create_bitmap()
        self.GetParent().check_size(bitmap.GetSize())
        dc = wx.AutoBufferedPaintDC(self)
        dc.DrawBitmap(bitmap, 0, 0)

class ControlPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.auto_button = wx.CheckBox(self, -1, "Auto")
        self.onoff_button = wx.CheckBox(self, -1, "On/Off")
        self.sizer.AddStretchSpacer()
        self.sizer.Add(self.auto_button, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.sizer.Add(self.onoff_button, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.SetSizerAndFit(self.sizer)
        self.SetMinSize((-1, 20))

class Frame(wx.Frame):
    def __init__(self, cl_model):
        style = wx.DEFAULT_FRAME_STYLE & ~wx.RESIZE_BORDER & ~wx.MAXIMIZE_BOX
        super(Frame, self).__init__(None, -1, 'OpenCL Mandelbrot', style=style)
        self.cl_model = cl_model
        
        self.controlpanel = ControlPanel(self)
        self.controlpanel.auto_button.Bind(wx.EVT_CHECKBOX, self.OnButton)
        self.controlpanel.onoff_button.Bind(wx.EVT_CHECKBOX, self.OnButton)        
        self.mainsizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        font = ImageFont.truetype("arial.ttf", 20)
        self.panel = Panel(self, self.cl_model, font)
        self.sizer.Add(self.panel, 1, wx.EXPAND)
        self.mainsizer.Add(self.sizer, 1, wx.EXPAND)
        self.mainsizer.Add(self.controlpanel, 0, wx.EXPAND)
        self.SetSizerAndFit(self.mainsizer)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def check_size(self, size):
        w,h = size
        size = w * 1, h
        if self.GetClientSize() != size:
            self.SetClientSize(size)
            self.Center()

    def OnClose(self, event):
        self.panel.Destroy()
        event.Skip()

    def OnButton(self, event):
        event.Skip()
        
        
        
def main():    

    cl_model = cl_unit()
    
    # now start the wx GUI
    app = wx.App(None)
    frame = Frame(cl_model = cl_model)
    frame.Center()
    frame.Show()

    app.MainLoop()

if __name__ == '__main__':    
    main()
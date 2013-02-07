//
//  View.m
//  CLTest
//
//  Created by Michael Fogleman on 2/5/13.
//  Copyright (c) 2013 Michael Fogleman. All rights reserved.
//

#import "View.h"
#include <OpenCL/opencl.h>
#include "mandelbrot.cl.h"

#define USE_CL 1
#define USE_JULIA 1

#define W 1024
#define H 768
#define N (W * H)
#define MAX_ITERATIONS 256
#define ZOOM 1.5f

#define RAINBOW 0
#define GREEN 1
#define GREEN_INVERTED 2
#define BLUE 3
#define BLUE_INVERTED 4
#define COLOR_MODES 5

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

@implementation View

- (void)mandelbrot { // plain C implementation
    for (int index = 0; index < N; index++) {
        float result;
        float i = index % W;
        float j = index / W;
        float x0 = _x + _w * (i / W);
        float y0 = _y + _h * (j / H);
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
        _mem[index] = result;
    }
}

- (void)update {
    NSDate *start = [NSDate date];
    if (USE_CL) {
        dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
        if (!queue) {
            queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
        }
        dispatch_sync(queue, ^{
            cl_ndrange range = {1, {0, 0, 0}, {N, 0, 0}, {0, 0, 0}};
            if (USE_JULIA) {
                julia_kernel(&range, self.max, W, H, self.x, self.y, self.w, self.h, self.xc, self.yc, self.cl_mem);
            }
            else {
                mandelbrot_kernel(&range, self.max, W, H, self.x, self.y, self.w, self.h, self.cl_mem);
            }
            gcl_memcpy(self.mem, self.cl_mem, sizeof(cl_float) * N);
        });
    }
    else {
        [self mandelbrot];
    }
    NSDate *end = [NSDate date];
    self.elapsed = [end timeIntervalSinceDate:start];
    [self setNeedsDisplay:YES];
}

- (id)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        self.mem = malloc(sizeof(cl_float) * N);
        self.cl_mem = gcl_malloc(sizeof(cl_float) * N, NULL, CL_MEM_WRITE_ONLY);
        self.max = MAX_ITERATIONS;
        if (USE_JULIA) {
            self.x = -2.0f;
            self.y = -1.5f;
        }
        else {
            self.x = -2.5f;
            self.y = -1.5f;
        }
        self.w = 4;
        self.h = 3;
        self.xc = -0.8f;
        self.yc = 0.156f;
        self.elapsed = 0;
        self.color_mode = GREEN;
        [self update];
    }
    return self;
}

- (void)dealloc {
    gcl_free(self.cl_mem);
    free(self.mem);
}

- (BOOL)isFlipped {
    return YES;
}

- (void)drawRect:(NSRect)dirtyRect {
    // compute range
    int hi = self.max;
    int lo = self.mem[0];
    for (int index = 1; index < N; index++) {
        if (self.mem[index] < lo) {
            lo = self.mem[index];
        }
    }
    int diff = hi - lo;
    // create image data
    unsigned char *data = malloc(sizeof(int) * N * 4);
    int i = 0;
    for (int index = 0; index < N; index++) {
        float r, g, b;
        if (self.mem[index] >= hi) {
            r = g = b = 0;
        }
        else {
            float p = logf(self.mem[index] - lo + 1) / logf(diff + 1);
            float h, s, v;
            switch (self.color_mode) {
                case GREEN:
                    h = 120;
                    s = 1;
                    v = 1 - p;
                    break;
                case GREEN_INVERTED:
                    h = 120;
                    s = 1;
                    v = p;
                    break;
                case BLUE:
                    h = 220;
                    s = 1;
                    v = 1 - p;
                    break;
                case BLUE_INVERTED:
                    h = 220;
                    s = 1;
                    v = p;
                    break;
                default: // RAINBOW
                    h = p * 360;
                    s = 0.75;
                    v = 1;
                    break;
            }
            hsv_to_rgb(&r, &g, &b, h, s, v);
        }
        data[i++] = r * 255;
        data[i++] = g * 255;
        data[i++] = b * 255;
        data[i++] = 255;
    }
    // create image from data
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef bitmapContext = CGBitmapContextCreate(data, W, H, 8, W * 4, colorSpace, kCGBitmapByteOrderDefault | kCGImageAlphaPremultipliedLast);
    CGImageRef cgImage = CGBitmapContextCreateImage(bitmapContext);
    NSImage *image = [[NSImage alloc] initWithCGImage:cgImage size:NSZeroSize];
    CGImageRelease(cgImage);
    CGContextRelease(bitmapContext);
    CFRelease(colorSpace);
    free(data);
    // draw image
    [image drawInRect:self.bounds fromRect:self.bounds operation:NSCompositeSourceOver fraction:1 respectFlipped:YES hints:nil];
    NSString *text = [NSString stringWithFormat:@"(%.3f, %.3f) (%.3f x %.3f) %.3f sec.", self.x, self.y, self.w, self.h, self.elapsed];
    NSFont *font = [NSFont fontWithName:@"Helvetica" size:18];
    NSDictionary *attrs = [NSDictionary dictionaryWithObjectsAndKeys:font, NSFontAttributeName, nil];
    [text drawAtPoint:NSMakePoint(10, 10) withAttributes:attrs];
}

- (void)mouseDown:(NSEvent *)event {
    NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
    if (USE_JULIA) {
        self.xc = 2 * point.x / self.bounds.size.width - 1;
        self.yc = 2 * point.y / self.bounds.size.height - 1;
    }
    else {
        float x = self.x + self.w * point.x / W;
        float y = self.y + self.h * point.y / H;
        self.w /= ZOOM;
        self.h /= ZOOM;
        self.x = x - self.w / 2;
        self.y = y - self.h / 2;
    }
    [self update];
}

- (void)mouseDragged:(NSEvent *)event {
    if (USE_JULIA) {
        NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
        self.xc = 2 * point.x / self.bounds.size.width - 1;
        self.yc = 2 * point.y / self.bounds.size.height - 1;
        [self update];
    }
}

- (void)rightMouseDown:(NSEvent *)event {
    NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
    float x = self.x + self.w * point.x / W;
    float y = self.y + self.h * point.y / H;
    self.w *= ZOOM;
    self.h *= ZOOM;
    self.x = x - self.w / 2;
    self.y = y - self.h / 2;
    [self update];
}

- (IBAction)onZoomIn:(id)sender {
    float x = self.x + self.w / 2;
    float y = self.y + self.h / 2;
    self.w /= ZOOM;
    self.h /= ZOOM;
    self.x = x - self.w / 2;
    self.y = y - self.h / 2;
    [self update];
}

- (IBAction)onZoomOut:(id)sender {
    float x = self.x + self.w / 2;
    float y = self.y + self.h / 2;
    self.w *= ZOOM;
    self.h *= ZOOM;
    self.x = x - self.w / 2;
    self.y = y - self.h / 2;
    [self update];
}

- (IBAction)onPanLeft:(id)sender {
    self.x -= self.w / 4;
    [self update];
}

- (IBAction)onPanRight:(id)sender {
    self.x += self.w / 4;
    [self update];
}

- (IBAction)onPanUp:(id)sender {
    self.y -= self.h / 4;
    [self update];
}

- (IBAction)onPanDown:(id)sender {
    self.y += self.h / 4;
    [self update];
}

- (IBAction)onColorMode:(id)sender {
    self.color_mode = (self.color_mode + 1) % COLOR_MODES;
    [self setNeedsDisplay:YES];
}

@end

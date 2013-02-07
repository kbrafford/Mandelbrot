//
//  View.h
//  CLTest
//
//  Created by Michael Fogleman on 2/5/13.
//  Copyright (c) 2013 Michael Fogleman. All rights reserved.
//

#import <Cocoa/Cocoa.h>

@interface View : NSView

@property (assign, nonatomic) float *mem;
@property (assign, nonatomic) void *cl_mem;
@property (assign, nonatomic) float max;
@property (assign, nonatomic) float x;
@property (assign, nonatomic) float y;
@property (assign, nonatomic) float w;
@property (assign, nonatomic) float h;
@property (assign, nonatomic) float xc;
@property (assign, nonatomic) float yc;
@property (assign, nonatomic) float elapsed;
@property (assign, nonatomic) int color_mode;

@end

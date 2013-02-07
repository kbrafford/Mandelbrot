kernel void mandelbrot(
    int _max, int _xs, int _ys,
    float _x, float _y, float _w, float _h,
    global float *output)
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

kernel void julia(
    int _max, int _xs, int _ys,
    float _x, float _y, float _w, float _h,
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

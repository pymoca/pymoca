connector HQ
    Real H;
    flow Real Q;
end HQ;

model Channel
	HQ up;
	HQ down;
equation
	connect(up, down);
end Channel;

model HBC
	HQ up;
equation
	up.H = 0;
end HBC;

model QBC
	HQ down;
equation
	down.Q = 0;
end QBC;

model System
	Channel a;
	Channel b;
	Channel c;
	QBC qa;
	HQ p;
	HBC hb;
	HQ zerotest;
equation
    p.Q = 0;
	connect(qa.down, a.up);
	connect(p, c.up);
	connect(a.down, b.up);
	connect(c.down, b.up);
	connect(b.down, hb.up);
end System;
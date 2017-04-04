connector HQ
    Real H;
    flow Real Q;
end HQ;

model Channel
	HQ up;
	HQ down;
equation
	up.H = down.H;
	up.Q + down.Q = 0;
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
	QBC qc;
	HBC hb;
equation
	connect(qa.down, a.up);
	connect(qc.down, c.up)
	connect(a.down, b.up);
	connect(c.down, b.up);
	connect(b.down, hb.up);
end System;

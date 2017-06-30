connector HQBase
    Real H;
    flow Real Q;
end HQBase;

connector HQZ
	extends HQBase;
    Real Z;
end HQZ;

model ChannelZ
	replaceable connector port = HQBase;
	port up;
	HQZ down;
equation
	up.H = down.H;
	up.Q + down.Q = 0;
end ChannelZ;

model HBCZ
	HQZ up;
equation
	up.H = 0;
end HBCZ;

model QBCZ
	HQZ down;
equation
	down.Q = 0;
end QBCZ;

model SystemZ
	ChannelZ a(redeclare connector port = HQZ);
	ChannelZ b(redeclare connector port = HQZ);
	ChannelZ c(redeclare connector port = HQZ);
	ChannelZ d;
	QBCZ qa;
	QBCZ qc;
	HBCZ hb;
equation
	connect(qa.down, a.up);
	connect(qc.down, c.up);
	connect(a.down, b.up);
	connect(c.down, b.up);
	connect(b.down, hb.up);
end SystemZ;

OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4551[3];
rz(4.3889776) q[0];
sx q[0];
rz(4.3788248) q[0];
sx q[0];
rz(14.653435) q[0];
rz(3.0115549) q[1];
sx q[1];
rz(-0.16133134) q[1];
sx q[1];
rz(2.370585) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
rz(3.5217041) q[2];
sx q[2];
rz(4.3676514) q[2];
sx q[2];
rz(15.201432) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(1.0545286) q[0];
sx q[0];
rz(-1.9043605) q[0];
sx q[0];
rz(1.8942077) q[0];
rz(0.50653144) q[1];
sx q[1];
rz(1.9155339) q[1];
sx q[1];
rz(2.370585) q[2];
sx q[2];
rz(-2.9802613) q[2];
sx q[2];
rz(0.13003778) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4551[0];
measure q[1] -> c4551[1];
measure q[2] -> c4551[2];

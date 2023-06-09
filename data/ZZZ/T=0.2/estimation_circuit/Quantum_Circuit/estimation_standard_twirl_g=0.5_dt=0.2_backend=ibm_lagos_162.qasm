OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4662[3];
rz(4.0648711) q[0];
sx q[0];
rz(3.2674907) q[0];
sx q[0];
rz(11.841133) q[0];
rz(1.5070613) q[1];
sx q[1];
rz(4.6042135) q[1];
sx q[1];
rz(13.789648) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
rz(-0.57724707) q[2];
sx q[2];
rz(-0.47434088) q[2];
sx q[2];
rz(1.1132435) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.72523743) q[0];
sx q[0];
rz(-0.12589802) q[0];
sx q[0];
rz(-0.92327848) q[0];
rz(-1.1132435) q[1];
sx q[1];
rz(0.47434088) q[1];
sx q[1];
rz(1.2232774) q[2];
sx q[2];
rz(-1.4626208) q[2];
sx q[2];
rz(1.6345313) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4662[0];
measure q[1] -> c4662[1];
measure q[2] -> c4662[2];

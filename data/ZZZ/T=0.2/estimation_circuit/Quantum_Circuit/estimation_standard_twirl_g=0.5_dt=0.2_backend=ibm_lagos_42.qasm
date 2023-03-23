OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4542[3];
rz(0.87185742) q[0];
sx q[0];
rz(-2.8119909) q[0];
sx q[0];
rz(1.4947448) q[0];
rz(1.5795274) q[1];
sx q[1];
rz(-1.260253) q[1];
sx q[1];
rz(2.7130561) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
rz(2.3261312) q[2];
sx q[2];
rz(3.3503408) q[2];
sx q[2];
rz(12.097292) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
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
rz(1.6468479) q[0];
sx q[0];
rz(-2.8119909) q[0];
sx q[0];
rz(-0.87185742) q[0];
rz(0.46907889) q[1];
sx q[1];
rz(2.9328445) q[1];
sx q[1];
rz(-0.42853653) q[2];
sx q[2];
rz(-1.8813396) q[2];
sx q[2];
rz(1.5620653) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4542[0];
measure q[1] -> c4542[1];
measure q[2] -> c4542[2];

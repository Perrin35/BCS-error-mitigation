OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4648[3];
rz(-2.24794) q[0];
sx q[0];
rz(-2.6139503) q[0];
sx q[0];
rz(-3.0406106) q[0];
rz(2.2987398) q[1];
sx q[1];
rz(3.7767435) q[1];
sx q[1];
rz(12.828765) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-1.5529288) q[2];
sx q[2];
rz(-2.8255415) q[2];
sx q[2];
rz(-0.27889859) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
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
rz(3.0406106) q[0];
sx q[0];
rz(-2.6139503) q[0];
sx q[0];
rz(2.24794) q[0];
rz(-2.8626941) q[1];
sx q[1];
rz(2.8255415) q[1];
sx q[1];
rz(-0.26239458) q[2];
sx q[2];
rz(-2.5064418) q[2];
sx q[2];
rz(-2.2987398) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4648[0];
measure q[1] -> c4648[1];
measure q[2] -> c4648[2];
OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4618[3];
rz(0.51772718) q[0];
sx q[0];
rz(6.2347925) q[0];
sx q[0];
rz(10.539948) q[0];
rz(-0.67767225) q[1];
sx q[1];
rz(-2.3723205) q[1];
sx q[1];
rz(2.7881907) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-0.98823524) q[2];
sx q[2];
rz(-2.7209644) q[2];
sx q[2];
rz(-1.0198332) q[2];
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
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
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
rz(-2.026423) q[0];
sx q[0];
rz(-3.0931998) q[0];
sx q[0];
rz(2.6238655) q[0];
rz(1.0198332) q[1];
sx q[1];
rz(-2.7209644) q[1];
sx q[1];
rz(-0.35340193) q[2];
sx q[2];
rz(-0.76927216) q[2];
sx q[2];
rz(-2.4639204) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4618[0];
measure q[1] -> c4618[1];
measure q[2] -> c4618[2];

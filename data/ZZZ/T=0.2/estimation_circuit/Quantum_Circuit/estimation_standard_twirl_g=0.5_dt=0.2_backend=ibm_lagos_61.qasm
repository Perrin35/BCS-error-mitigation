OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4561[3];
rz(5.6467747) q[0];
sx q[0];
rz(4.5949538) q[0];
sx q[0];
rz(15.684648) q[0];
rz(0.44879455) q[1];
sx q[1];
rz(-2.0754508) q[1];
sx q[1];
rz(1.3720472) q[1];
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
rz(-pi) q[1];
rz(3.3501292) q[2];
sx q[2];
rz(4.4698233) q[2];
sx q[2];
rz(12.738788) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.023314936) q[0];
sx q[0];
rz(-1.4533612) q[0];
sx q[0];
rz(-2.5051821) q[0];
rz(-2.9691755) q[1];
sx q[1];
rz(-1.3282307) q[1];
sx q[1];
rz(-1.3720472) q[2];
sx q[2];
rz(-2.0754508) q[2];
sx q[2];
rz(-0.44879455) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4561[0];
measure q[1] -> c4561[1];
measure q[2] -> c4561[2];

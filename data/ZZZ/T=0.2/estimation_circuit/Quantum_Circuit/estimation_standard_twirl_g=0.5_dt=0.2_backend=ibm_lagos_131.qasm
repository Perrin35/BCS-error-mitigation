OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4631[3];
rz(1.5284434) q[0];
sx q[0];
rz(3.6917124) q[0];
sx q[0];
rz(15.245507) q[0];
rz(1.3344517) q[1];
sx q[1];
rz(4.6987459) q[1];
sx q[1];
rz(10.422987) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
rz(-0.15863967) q[2];
sx q[2];
rz(-2.868395) q[2];
sx q[2];
rz(0.5006234) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
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
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-2.6791361) q[0];
sx q[0];
rz(-2.5914729) q[0];
sx q[0];
rz(-1.5284434) q[0];
rz(-0.5006234) q[1];
sx q[1];
rz(-2.868395) q[1];
sx q[1];
rz(-2.1433832) q[2];
sx q[2];
rz(-1.5571532) q[2];
sx q[2];
rz(1.8071409) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4631[0];
measure q[1] -> c4631[1];
measure q[2] -> c4631[2];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4585[3];
rz(4.2102701) q[0];
sx q[0];
rz(5.5877494) q[0];
sx q[0];
rz(13.148894) q[0];
rz(0.19470463) q[1];
sx q[1];
rz(-1.6912926) q[1];
sx q[1];
rz(-1.3897151) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
rz(-0.45721876) q[2];
sx q[2];
rz(-0.99605153) q[2];
sx q[2];
rz(2.9371052) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.5590693) q[0];
sx q[0];
rz(-0.69543586) q[0];
sx q[0];
rz(2.0729152) q[0];
rz(0.20448748) q[1];
sx q[1];
rz(0.99605153) q[1];
sx q[1];
rz(1.7518775) q[2];
sx q[2];
rz(-1.4503001) q[2];
sx q[2];
rz(2.946888) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4585[0];
measure q[1] -> c4585[1];
measure q[2] -> c4585[2];

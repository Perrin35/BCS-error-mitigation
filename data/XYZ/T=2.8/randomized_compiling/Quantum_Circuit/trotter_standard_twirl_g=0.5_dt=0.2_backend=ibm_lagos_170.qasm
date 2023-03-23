OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[3];
rz(-pi) q[0];
sx q[0];
rz(0.42975437) q[0];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
sx q[2];
rz(2.7118383) q[2];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.1076565) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.6339361) q[1];
sx q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-1.5339362) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.6076566) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.96015125) q[1];
sx q[1];
rz(-0.9601513) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(0.96015125) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-2.1814414) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.93775089) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.4038418) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-1.7038418) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.4377509) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(0.011439483) q[1];
sx q[1];
rz(-pi) q[1];
rz(-3.1301532) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-0.011439483) q[1];
rz(pi/2) q[2];
sx q[2];
rz(3.1301532) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.8592959) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-1.6822969) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-1.3592959) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.7822969) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.804615) q[1];
sx q[1];
rz(2.804615) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-2.804615) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.33697765) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
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
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-3.1132988) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-0.77170615) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-2.6698865) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.47170615) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.3896059) q[1];
sx q[1];
rz(-pi) q[1];
rz(1.389606) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(1.7519868) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.7519867) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
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
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.277513) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.877513) q[1];
sx q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-1.7775131) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.3640797) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.1061777) q[1];
sx q[1];
rz(2.1061777) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-2.1061777) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.035415) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.84803802) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.4935546) q[1];
sx q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(1.348038) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.3480381) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.1307776) q[1];
sx q[1];
rz(1.1307777) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
sx q[2];
rz(-0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-1.1307776) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.1307777) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(0.92588571) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(0.52588575) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(2.7157069) q[0];
rz(-pi) q[1];
sx q[1];
rz(2.7157069) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.7396055) q[1];
sx q[1];
rz(-1.7396055) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(1.7396055) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.7396055) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
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
rz(pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.5597966) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.7597966) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-2.0597966) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.0817961) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.1708134) q[1];
sx q[1];
rz(-pi) q[1];
rz(-0.97077925) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(0.9707793) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.97077925) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.65674335) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(2.0848494) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-1.9848493) q[0];
rz(-pi) q[1];
sx q[1];
rz(-1.9848494) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.2752189) q[1];
sx q[1];
rz(-pi) q[1];
rz(0.86637375) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(0.1) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-0.86637373) q[1];
rz(pi/2) q[2];
sx q[2];
rz(2.2752189) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
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
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.0253401) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.3162525) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-1.6162526) q[0];
sx q[1];
rz(-1.6162525) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.6857243) q[1];
sx q[1];
rz(-0.45586836) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-2.6857243) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-2.6857243) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
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
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-3.0715563) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-0.32996365) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(0.42996361) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.42996365) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.4450343) q[1];
sx q[1];
rz(-0.6965584) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-2.4450343) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.6965584) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
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
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.3902568) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-1.5513359) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(1.2513359) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.2513359) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.1807682) q[1];
sx q[1];
rz(-pi) q[1];
rz(0.96082448) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(2.1807682) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-0.96082448) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.6093927) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.2093927) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(1.0322) q[0];
sx q[1];
rz(1.0322) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.577748) q[1];
sx q[1];
rz(-1.5777481) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-1.5638447) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.5638446) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
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
rz(-pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.3988916) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.5427011) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-1.8988916) q[0];
rz(-pi) q[1];
sx q[1];
rz(-1.8988916) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.2709048) q[1];
sx q[1];
rz(-1.8706878) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.8706879) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.2709049) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];

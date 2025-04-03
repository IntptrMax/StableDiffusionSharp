using StableDiffusionSharp;
using System.Diagnostics;

namespace StableDiffusionDemo_Winform
{
	public partial class FormMain : Form
	{
		string modelPath = string.Empty;
		string vaeModelPath = string.Empty;
		StableDiffusion? sd;

		public FormMain()
		{
			InitializeComponent();
		}

		private void FormMain_Load(object sender, EventArgs e)
		{
			ComboBox_Device.SelectedIndex = 0;
			ComboBox_Precition.SelectedIndex = 0;
		}

		private void Button_ModelScan_Click(object sender, EventArgs e)
		{
			FileDialog fileDialog = new OpenFileDialog();
			fileDialog.Filter = "Model files|*.safetensors;*.ckpt;*.pt;*.pth|All files|*.*";
			if (fileDialog.ShowDialog() == DialogResult.OK)
			{
				TextBox_ModelPath.Text = fileDialog.FileName;
				modelPath = fileDialog.FileName;
			}
		}

		private void Button_ModelLoad_Click(object sender, EventArgs e)
		{
			if (File.Exists(modelPath))
			{
				SDDeviceType deviceType = ComboBox_Device.SelectedIndex == 0 ? SDDeviceType.CUDA : SDDeviceType.CPU;
				SDScalarType scalarType = ComboBox_Precition.SelectedIndex == 0 ? SDScalarType.Float16 : SDScalarType.Float32;
				Task.Run(() =>
				{
					base.Invoke(() => Button_ModelLoad.Enabled = false);
					sd = new StableDiffusion(deviceType, scalarType);
					sd.StepProgress += Sd_StepProgress;
					sd.LoadModel(modelPath, vaeModelPath);
					base.Invoke(() =>
					{
						Button_ModelLoad.Enabled = true;
						Button_Generate.Enabled = true;
						Label_State.Text = "Model loaded.";
					});
				});
			}
		}

		private void Button_VAEModelScan_Click(object sender, EventArgs e)
		{
			FileDialog fileDialog = new OpenFileDialog();
			fileDialog.Filter = "Model files|*.safetensors;*.ckpt;*.pt;*.pth|All files|*.*";
			if (fileDialog.ShowDialog() == DialogResult.OK)
			{
				TextBox_VaePath.Text = fileDialog.FileName;
				vaeModelPath = fileDialog.FileName;
			}
		}

		private void Sd_StepProgress(object? sender, StableDiffusion.StepEventArgs e)
		{
			base.Invoke(() =>
			{
				Label_State.Text = $"Progress: {e.CurrentStep}/{e.TotalSteps}";
				if (e.VaeApproxImg != null)
				{
					MemoryStream memoryStream = new MemoryStream();
					e.VaeApproxImg.Write(memoryStream, ImageMagick.MagickFormat.Jpg);
					base.Invoke(() =>
					{
						PictureBox_Output.Image = Image.FromStream(memoryStream);
					});
				}
			});
		}

		private void Button_Generate_Click(object sender, EventArgs e)
		{
			string prompt = TextBox_Prompt.Text;
			string nprompt = TextBox_NPrompt.Text;
			int step = (int)NumericUpDown_Step.Value;
			float cfg = (float)NumericUpDown_CFG.Value;
			long seed = 0;
			int width = (int)NumericUpDown_Width.Value;
			int height = (int)NumericUpDown_Height.Value;
			int clipSkip = (int)NumericUpDown_ClipSkip.Value;

			Task.Run(() =>
			{
				Stopwatch stopwatch = Stopwatch.StartNew();
				base.Invoke(() =>
				{
					Button_ModelLoad.Enabled = false;
					Button_Generate.Enabled = false;
					Label_State.Text = "Generating...";
				});
				ImageMagick.MagickImage image = sd.TextToImage(prompt, nprompt, clipSkip, width, height, step, seed, cfg);
				MemoryStream memoryStream = new MemoryStream();
				image.Write(memoryStream, ImageMagick.MagickFormat.Jpg);
				base.Invoke(() =>
				{
					PictureBox_Output.Image = Image.FromStream(memoryStream);
					Button_ModelLoad.Enabled = true;
					Button_Generate.Enabled = true;
					Label_State.Text = $"Done. It takes {stopwatch.Elapsed.TotalSeconds.ToString("f2")} s";
				});
				GC.Collect();
			});
		}
	}
}
